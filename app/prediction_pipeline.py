"""
League-wide gradient-boosted prediction pipeline for the Prediction tab.

Trains calibrated tree models on full Trackman pitch pools, engineers rich features,
applies smoothed batter/pitcher priors, scores recommendations (command / put-away / danger),
and produces short explanations. Optional SHAP; lightweight fallback if unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Suppress sklearn convergence warnings in dashboard context
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config: scoring weights (tune without touching algorithm code)
# ---------------------------------------------------------------------------
PITCH_TYPE_COL_DEFAULT = "TaggedPitchType"

ML_HARD_CONTACT_EV = 80.0
ML_MIN_TRAIN_ROWS = 200 # league-level minimum rows to train (dashboard-friendly)
ML_MIN_PER_CLASS = 25
ML_MAX_TRAIN_ROWS = 120_000  # subsample for latency if league dump is huge
PRIOR_K = 25.0  # empirical-Bayes-style smoothing strength for rate features
ML_LOW_SAMPLE_ML_WARN = "Low sample — estimates lean on league priors"

# Composite score weights (higher = better for attack/putaway; danger is separate)
SCORE_WEIGHT_ATTACK_STRIKE = 1.15
SCORE_WEIGHT_ATTACK_HARD_PENALTY = 1.35
SCORE_WEIGHT_ATTACK_STRIKE_CAL = 0.25  # mild preference toward ~65% strike prob (command)

SCORE_WEIGHT_PUT_WHIFF = 1.25
SCORE_WEIGHT_PUT_HARD_PENALTY = 1.45
SCORE_WEIGHT_PUT_TWOSTRIKE_BOOST = 0.35  # extra value when simulating 0-2 count

SCORE_WEIGHT_DANGER_HARD = 1.35
SCORE_WEIGHT_DANGER_STRIKE = 0.85  # unreliable strike = worse caution story

PREDICTION_MIN_STRIKE_N = 12
PREDICTION_MIN_SWINGS_PUTAWAY = 8
PREDICTION_MIN_CONTACT_CAUTION = 5

MODEL_CACHE: dict[str, Any] = {}
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".prediction_model_cache.pkl")


def clear_prediction_cache(remove_disk: bool = True) -> None:
    """Clear in-memory cache and optionally remove persisted model cache."""
    MODEL_CACHE.clear()
    if remove_disk:
        try:
            if os.path.isfile(CACHE_PATH):
                os.remove(CACHE_PATH)
        except Exception:
            # Best effort only; app can continue with in-memory retrain.
            pass


def _valid_pitch_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return (
        series.notna()
        & s.ne("")
        & s.ne("undefined")
        & s.ne("other")
        & s.ne("nan")
    )


def _league_fingerprint(league_df: pd.DataFrame) -> str:
    if league_df is None or league_df.empty:
        return "empty"
    n = len(league_df)
    parts = [str(n)]
    if "Date" in league_df.columns:
        try:
            d = pd.to_datetime(league_df["Date"], errors="coerce")
            parts.append(str(d.min()))
            parts.append(str(d.max()))
        except Exception:
            pass
    raw = "|".join(parts).encode()
    return hashlib.md5(raw).hexdigest()[:16]


def smooth_rate(count: float, rate: float, global_rate: float, k: float = PRIOR_K) -> float:
    """Shrinkage toward league/pitch-type mean for stable player features."""
    c = max(0.0, float(count))
    g = float(global_rate) if pd.notna(global_rate) else 0.0
    r = float(rate) if pd.notna(rate) else g
    if c + k <= 0:
        return g
    return (c * r + k * g) / (c + k)


def _same_hand(p_throw: str, b_side: str) -> bool:
    pt = str(p_throw).strip().lower()[:1]
    bs = str(b_side).strip().lower()[:1]
    if pt not in ("r", "l") or bs not in ("r", "l"):
        return False
    return pt == bs


def engineer_pitch_features(
    df: pd.DataFrame,
    pitch_col: str,
    zone_lr: tuple[float, float, float, float],
) -> pd.DataFrame:
    """
    Pitch-level feature engineering: lags, zone geometry, count buckets, platoon.
    zone_lr = (ZONE_LEFT, ZONE_RIGHT, ZONE_BOTTOM, ZONE_TOP)
    """
    if df is None or df.empty:
        return df
    d = df.loc[_valid_pitch_mask(df[pitch_col])].copy()
    if d.empty:
        return d

    zl, zr, zb, zt = zone_lr
    mid_h = (zb + zt) / 2.0
    mid_s = (zl + zr) / 2.0

    # Sort for stable lag features within plate appearance
    sort_cols = [c for c in ["Date", "Time", "PitcherId", "BatterId", "PitchofPA"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols, kind="mergesort")

    grp_keys = [c for c in ["Date", "PitcherId", "BatterId"] if c in d.columns]
    if grp_keys:
        g = d.groupby(grp_keys, sort=False)
        d["prev_pitch_type"] = g[pitch_col].shift(1)
        if "RelSpeed" in d.columns:
            d["prev_RelSpeed"] = g["RelSpeed"].shift(1)
        else:
            d["prev_RelSpeed"] = np.nan
    else:
        d["prev_pitch_type"] = np.nan
        d["prev_RelSpeed"] = np.nan

    d["prev_pitch_type"] = d["prev_pitch_type"].fillna("__FIRST__")
    for c in [
        "RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak",
        "PlateLocSide", "PlateLocHeight", "Balls", "Strikes", "PitchofPA",
    ]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "RelSpeed" in d.columns and "prev_RelSpeed" in d.columns:
        d["delta_RelSpeed"] = d["RelSpeed"] - d["prev_RelSpeed"]
    else:
        d["delta_RelSpeed"] = np.nan

    if "PlateLocSide" in d.columns and "PlateLocHeight" in d.columns:
        side = d["PlateLocSide"]
        height = d["PlateLocHeight"]
        loc_ok = side.notna() & height.notna()
        d["in_zone"] = (
            loc_ok & side.between(zl, zr, inclusive="both") & height.between(zb, zt, inclusive="both")
        ).astype(float)
        d["zone_upper"] = (loc_ok & (height > mid_h)).astype(float)
        # Inner/outer vs batter box: mirror for LHB
        if "BatterSide" in d.columns:
            bs = d["BatterSide"].astype(str).str.strip().str.lower().str[:1]
            signed = side.copy()
            signed = signed.where(bs.ne("l"), -signed)
            d["zone_inner"] = (loc_ok & (signed < mid_s)).astype(float)
        else:
            d["zone_inner"] = (loc_ok & (side < mid_s)).astype(float)
    else:
        d["in_zone"] = np.nan
        d["zone_upper"] = np.nan
        d["zone_inner"] = np.nan

    if "Strikes" in d.columns:
        st = d["Strikes"].fillna(0).clip(lower=0, upper=2)
        d["is_two_strike"] = (st >= 2).astype(float)
    else:
        d["is_two_strike"] = 0.0

    if "Balls" in d.columns and "Strikes" in d.columns:
        b = d["Balls"].fillna(0).clip(0, 3)
        st = d["Strikes"].fillna(0).clip(0, 2)
        hitter = ((b - st) >= 2) & (b >= 2)
        pitcher = (st == 2) & (b <= 1)
        d["is_hitter_count"] = hitter.astype(float)
        d["is_pitcher_count"] = pitcher.astype(float)
    else:
        d["is_hitter_count"] = 0.0
        d["is_pitcher_count"] = 0.0

    if "PitcherThrows" in d.columns and "BatterSide" in d.columns:
        d["platoon_same"] = d.apply(
            lambda r: float(_same_hand(r.get("PitcherThrows", ""), r.get("BatterSide", ""))),
            axis=1,
        )
    else:
        d["platoon_same"] = 0.0

    return d


def _strike_whiff_hard_masks(pc: pd.Series, ev: pd.Series | None):
    pc = pc.astype(str).str.strip()
    strike_events = {
        "StrikeCalled", "StrikeSwinging", "FoulBallFieldable",
        "FoulBallNotFieldable", "InPlay",
    }
    y_strike = pc.isin(strike_events).astype(int)
    y_whiff = pc.eq("StrikeSwinging").astype(int)
    in_play = pc.eq("InPlay")
    if ev is None:
        ev = pd.Series(np.nan, index=pc.index)
    y_hard = (in_play & (ev >= ML_HARD_CONTACT_EV)).astype(int)
    return y_strike, y_whiff, y_hard


def compute_smoothed_priors(league_eng: pd.DataFrame, pitch_col: str) -> dict[str, pd.DataFrame]:
    """League / batter / pitcher rate tables with smoothing helpers."""
    pc = league_eng["PitchCall"].astype(str).str.strip()
    ev = pd.to_numeric(league_eng["ExitSpeed"], errors="coerce") if "ExitSpeed" in league_eng.columns else None
    if ev is None:
        ev = pd.Series(np.nan, index=league_eng.index)
    swing_m = pc.isin({"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"})
    contact_m = pc.isin({"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"})
    in_play = pc.eq("InPlay")
    hard_m = in_play & (ev >= ML_HARD_CONTACT_EV)
    inz = league_eng["in_zone"].fillna(0)
    out_zone = inz == 0
    chase_m = out_zone & swing_m

    league_eng = league_eng.copy()
    league_eng["_swing"] = swing_m.astype(int)
    league_eng["_whiff"] = pc.eq("StrikeSwinging").astype(int)
    league_eng["_hard"] = hard_m.astype(int)
    league_eng["_strike"] = pc.isin({
        "StrikeCalled", "StrikeSwinging", "FoulBallFieldable",
        "FoulBallNotFieldable", "InPlay",
    }).astype(int)
    league_eng["_contact"] = contact_m.astype(int)
    league_eng["_chase"] = chase_m.astype(int)

    g_pt = league_eng.groupby(pitch_col, dropna=False).agg(
        g_n=("PitchCall", "count"),
        g_whiff_rate=("_whiff", "mean"),
        g_hard_rate=("_hard", "mean"),
        g_strike_rate=("_strike", "mean"),
        g_chase_rate=("_chase", "mean"),
    ).reset_index()

    # Batter x pitch type
    if "BatterId" in league_eng.columns:
        b_pt = league_eng.groupby(["BatterId", pitch_col], dropna=False).agg(
            b_n=("PitchCall", "count"),
            b_whiff=("_whiff", "mean"),
            b_hard=("_hard", "mean"),
            b_chase=("_chase", "mean"),
        ).reset_index()
        b_pt = b_pt.merge(g_pt[[pitch_col, "g_whiff_rate", "g_hard_rate", "g_chase_rate"]], on=pitch_col, how="left")
        b_pt["b_whiff_s"] = b_pt.apply(
            lambda r: smooth_rate(r["b_n"], r["b_whiff"], r["g_whiff_rate"]), axis=1,
        )
        b_pt["b_hard_s"] = b_pt.apply(
            lambda r: smooth_rate(r["b_n"], r["b_hard"], r["g_hard_rate"]), axis=1,
        )
        b_pt["b_chase_s"] = b_pt.apply(
            lambda r: smooth_rate(r["b_n"], r["b_chase"], r["g_chase_rate"]), axis=1,
        )
    else:
        b_pt = pd.DataFrame()

    if "PitcherId" in league_eng.columns:
        p_pt = league_eng.groupby(["PitcherId", pitch_col], dropna=False).agg(
            p_n=("PitchCall", "count"),
            p_strike=("_strike", "mean"),
            p_whiff=("_whiff", "mean"),
            p_hard=("_hard", "mean"),
        ).reset_index()
        p_pt = p_pt.merge(g_pt[[pitch_col, "g_strike_rate", "g_whiff_rate", "g_hard_rate"]], on=pitch_col, how="left")
        p_pt["p_strike_s"] = p_pt.apply(
            lambda r: smooth_rate(r["p_n"], r["p_strike"], r["g_strike_rate"]), axis=1,
        )
        p_pt["p_whiff_s"] = p_pt.apply(
            lambda r: smooth_rate(r["p_n"], r["p_whiff"], r["g_whiff_rate"]), axis=1,
        )
        p_pt["p_hard_s"] = p_pt.apply(
            lambda r: smooth_rate(r["p_n"], r["p_hard"], r["g_hard_rate"]), axis=1,
        )
        # Usage within pitcher
        tot = league_eng.groupby("PitcherId")["PitchCall"].count().rename("tot").reset_index()
        p_pt = p_pt.merge(tot, on="PitcherId", how="left")
        p_pt["p_usage_s"] = p_pt["p_n"] / p_pt["tot"].replace(0, np.nan)
        g_usage = league_eng.groupby(pitch_col)["PitchCall"].count()
        g_usage = (g_usage / g_usage.sum()).rename("g_usage")
        p_pt = p_pt.merge(g_usage.reset_index(), on=pitch_col, how="left")
        p_pt["p_usage_sm"] = p_pt.apply(
            lambda r: smooth_rate(r["p_n"], r["p_usage_s"], r["g_usage"]), axis=1,
        )
    else:
        p_pt = pd.DataFrame()

    return {"global_ptype": g_pt, "batter_ptype": b_pt, "pitcher_ptype": p_pt}


def merge_priors(
    df: pd.DataFrame,
    priors: dict[str, pd.DataFrame],
    pitch_col: str,
) -> pd.DataFrame:
    out = df.copy()
    g = priors["global_ptype"][[pitch_col, "g_whiff_rate", "g_hard_rate", "g_chase_rate", "g_strike_rate"]].drop_duplicates()

    if not priors["batter_ptype"].empty and "BatterId" in out.columns:
        out = out.merge(
            priors["batter_ptype"][
                ["BatterId", pitch_col, "b_whiff_s", "b_hard_s", "b_chase_s"]
            ],
            on=["BatterId", pitch_col],
            how="left",
        )
    else:
        out["b_whiff_s"] = np.nan
        out["b_hard_s"] = np.nan
        out["b_chase_s"] = np.nan

    out = out.merge(g, on=pitch_col, how="left")
    for c, gc in [("b_whiff_s", "g_whiff_rate"), ("b_hard_s", "g_hard_rate"), ("b_chase_s", "g_chase_rate")]:
        out[c] = out[c].fillna(out[gc])

    if not priors["pitcher_ptype"].empty and "PitcherId" in out.columns:
        pp = priors["pitcher_ptype"][
            ["PitcherId", pitch_col, "p_strike_s", "p_whiff_s", "p_hard_s", "p_usage_sm"]
        ]
        out = out.merge(pp, on=["PitcherId", pitch_col], how="left")
    else:
        out["p_strike_s"] = np.nan
        out["p_whiff_s"] = np.nan
        out["p_hard_s"] = np.nan
        out["p_usage_sm"] = np.nan

    out["p_strike_s"] = out["p_strike_s"].fillna(out["g_strike_rate"])
    out["p_whiff_s"] = out["p_whiff_s"].fillna(out["g_whiff_rate"])
    out["p_hard_s"] = out["p_hard_s"].fillna(out["g_hard_rate"])
    out["p_usage_sm"] = out["p_usage_sm"].fillna(0.05)

    drop_g = [c for c in ["g_whiff_rate", "g_hard_rate", "g_chase_rate", "g_strike_rate"] if c in out.columns]
    out.drop(columns=drop_g, inplace=True, errors="ignore")
    return out


def filter_league_for_side(league_df: pd.DataFrame, batter_side_filter: str) -> pd.DataFrame:
    if league_df is None or league_df.empty or batter_side_filter in (None, "all"):
        return league_df
    if "BatterSide" not in league_df.columns:
        return league_df
    s = league_df["BatterSide"].astype(str).str.strip().str.lower().str[:1]
    if batter_side_filter == "right":
        return league_df[s.isin(["r", "right"])]
    if batter_side_filter == "left":
        return league_df[s.isin(["l", "left"])]
    return league_df


def _make_base_estimator():
    try:
        import xgboost as xgb  # type: ignore

        return xgb.XGBClassifier(
            n_estimators=220,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=0,
            tree_method="hist",
        )
    except Exception:
        pass
    try:
        import lightgbm as lgb  # type: ignore

        return lgb.LGBMClassifier(
            n_estimators=220,
            num_leaves=48,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=40,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
    except Exception:
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=10,
        max_iter=320,
        learning_rate=0.06,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )


def _build_sklearn_pipeline(num_cols: list[str], cat_cols: list[str], estimator):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        )
    if not transformers:
        return None
    prep = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prep", prep), ("clf", estimator)])


def _fit_calibrated(X: pd.DataFrame, y: pd.Series, num_cols: list[str], cat_cols: list[str]):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    y = pd.Series(y).astype(int)
    if y.nunique() < 2 or len(y) < ML_MIN_PER_CLASS * 2:
        return None, {}

    base = _make_base_estimator()
    pipe = _build_sklearn_pipeline(num_cols, cat_cols, base)
    if pipe is None:
        return None, {}

    try:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
    except ValueError:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )
    pipe.fit(X_tr, y_tr)
    metrics: dict[str, float] = {}
    try:
        proba = pipe.predict_proba(X_va)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_va, proba))
        metrics["pr_auc"] = float(average_precision_score(y_va, proba))
        metrics["log_loss"] = float(log_loss(y_va, proba, labels=[0, 1]))
    except Exception as e:
        logger.debug("validation metrics skipped: %s", e)

    cal = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
    cal.fit(X_va, y_va)
    return cal, metrics


def _get_feature_names(num_cols: list[str], cat_cols: list[str]) -> list[str]:
    return [f"num::{c}" for c in num_cols] + [f"cat::{c}" for c in cat_cols]


def fallback_explanation(
    row: pd.Series,
    baseline: pd.Series,
    feat_names: list[str],
    p_strike: float,
    p_hard: float,
    kind: str,
) -> str:
    """Template + top deviations vs baseline profile (no SHAP)."""
    deltas = []
    for c in feat_names:
        if "::" not in c:
            continue
        _, name = c.split("::", 1)
        if name not in row.index or name not in baseline.index:
            continue
        try:
            rv = float(row[name])
            bv = float(baseline[name])
            if pd.isna(rv) or pd.isna(bv):
                continue
            diff = rv - bv
            if abs(diff) < 1e-6:
                continue
            if name in ("PlateLocSide",):
                deltas.append(("glove-side bias" if diff < 0 else "arm-side bias", abs(diff)))
            elif name == "PlateLocHeight":
                deltas.append(("elevated location" if diff > 0 else "lower location", abs(diff)))
            elif name == "RelSpeed":
                deltas.append(("above-profile velocity" if diff > 0 else "below-profile velocity", abs(diff)))
            elif name == "InducedVertBreak":
                deltas.append(("more vertical break than usual" if diff > 0 else "less vertical break than usual", abs(diff)))
            elif name in ("in_zone",):
                deltas.append(("more often in-zone in this profile" if diff > 0 else "fewer in-zone pitches", abs(diff)))
            elif name == "is_two_strike":
                deltas.append(("two-strike context" if diff > 0 else "early-count context", abs(diff)))
        except Exception:
            continue
    deltas.sort(key=lambda x: -x[1])
    parts = [d[0] for d in deltas[:3]]

    if kind == "attack":
        head = "Strong command profile: "
        if parts:
            return head + "model favors this pitch when " + ", ".join(parts) + ", lifting strike probability while keeping hard-contact risk contained."
        return head + f"high model strike probability ({p_strike:.0%}) vs typical shape for this arsenal."
    if kind == "putaway":
        head = "Put-away fit: "
        if parts:
            return head + ", ".join(parts).capitalize() + " align with elevated whiff potential in two-strike counts for this matchup context."
        return head + "whiff probability peaks for this pitch type under two-strike simulation with acceptable damage risk."
    head = "Damage risk: "
    if parts:
        return head + ", ".join(parts).capitalize() + f" coincide with higher hard-contact risk (~{p_hard:.0%}) for this batter-side pool."
    return head + f"hard-contact risk is elevated (~{p_hard:.0%}) relative to other pitch types in this profile."


def _matchup_note(batter_side_filter: str) -> str:
    if batter_side_filter == "right":
        return "Matchup context: vs RHB pool (league priors + filters)."
    if batter_side_filter == "left":
        return "Matchup context: vs LHB pool (league priors + filters)."
    return "Matchup context: combined handedness (neutral platoon priors)."


@dataclass
class TrainedBundle:
    models: dict[str, Any]
    num_cols: list[str]
    cat_cols: list[str]
    metrics: dict[str, dict[str, float]]
    warnings: dict[str, str]
    fingerprint: str


class ConstantProbModel:
    """Fallback classifier when target class balance is too small to train trees."""

    def __init__(self, p: float):
        self.p = float(min(max(p, 0.0), 1.0))

    def predict_proba(self, X):
        n = len(X) if X is not None else 0
        p1 = np.full(n, self.p, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def train_or_load_bundle(league_eng: pd.DataFrame, pitch_col: str) -> TrainedBundle | None:
    fp = _league_fingerprint(league_eng)
    global MODEL_CACHE
    if MODEL_CACHE.get("fp") == fp and "bundle" in MODEL_CACHE:
        return MODEL_CACHE["bundle"]
    disk = load_bundle_from_disk(fp)
    if disk is not None:
        MODEL_CACHE["fp"] = fp
        MODEL_CACHE["bundle"] = disk
        return disk

    num_cols = [
        "RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak",
        "PlateLocSide", "PlateLocHeight", "Balls", "Strikes", "PitchofPA",
        "prev_RelSpeed", "delta_RelSpeed",
        "in_zone", "zone_upper", "zone_inner",
        "is_two_strike", "is_hitter_count", "is_pitcher_count", "platoon_same",
        "b_whiff_s", "b_hard_s", "b_chase_s",
        "p_strike_s", "p_whiff_s", "p_hard_s", "p_usage_sm",
    ]
    cat_cols = [pitch_col, "prev_pitch_type", "PitcherThrows", "BatterSide"]
    num_cols = [c for c in num_cols if c in league_eng.columns]
    cat_cols = [c for c in cat_cols if c in league_eng.columns]

    pc = league_eng["PitchCall"].astype(str).str.strip()
    ev = pd.to_numeric(league_eng["ExitSpeed"], errors="coerce") if "ExitSpeed" in league_eng.columns else None
    y_strike, y_whiff, y_hard = _strike_whiff_hard_masks(pc, ev)

    X = league_eng[num_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("missing")

    if len(X) > ML_MAX_TRAIN_ROWS:
        X = X.sample(ML_MAX_TRAIN_ROWS, random_state=42)
        y_strike = y_strike.loc[X.index]
        y_whiff = y_whiff.loc[X.index]
        y_hard = y_hard.loc[X.index]

    metrics_all: dict[str, dict[str, float]] = {}
    warnings_all: dict[str, str] = {}
    models: dict[str, Any] = {}

    for name, y in [("strike", y_strike), ("whiff", y_whiff), ("hard", y_hard)]:
        pos = int(y.sum())
        neg = int(len(y) - pos)
        if len(y) < 10 or pos == 0 or neg == 0:
            # Fully degenerate target; use prior probability estimate.
            prior = float(np.clip(y.mean() if len(y) else 0.0, 0.0, 1.0))
            m = ConstantProbModel(prior)
            met = {"fallback_prior": prior}
            warnings_all[name] = "Fallback prior model: target has a single class in training pool."
        elif pos < ML_MIN_PER_CLASS or neg < ML_MIN_PER_CLASS:
            # Sparse minority class; still avoid brittle overfit by using prior fallback.
            prior = float(np.clip(y.mean(), 0.0, 1.0))
            m = ConstantProbModel(prior)
            met = {"fallback_prior": prior}
            warnings_all[name] = (
                f"Fallback prior model: class counts too small (pos={pos}, neg={neg})."
            )
        else:
            m, met = _fit_calibrated(X, y, num_cols, cat_cols)
            if m is None:
                prior = float(np.clip(y.mean(), 0.0, 1.0))
                m = ConstantProbModel(prior)
                met = {"fallback_prior": prior}
                warnings_all[name] = "Fallback prior model: boosted training failed."
        models[name] = m
        metrics_all[name] = met
        logger.info("prediction model %s metrics: %s", name, met)

    bundle = TrainedBundle(
        models=models,
        num_cols=num_cols,
        cat_cols=cat_cols,
        metrics=metrics_all,
        warnings=warnings_all,
        fingerprint=fp,
    )
    MODEL_CACHE["fp"] = fp
    MODEL_CACHE["bundle"] = bundle    # Optional disk cache (best-effort)
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(bundle, f)
    except Exception:
        pass

    return bundle


def load_bundle_from_disk(fingerprint: str) -> TrainedBundle | None:
    try:
        with open(CACHE_PATH, "rb") as f:
            b = pickle.load(f)
        if getattr(b, "fingerprint", None) == fingerprint:
            return b
    except Exception:
        pass
    return None


def _typical_pitch_profile_rows(
    pitcher_df: pd.DataFrame,
    pitch_col: str,
    count_spec: dict[str, int | float],
    mode_batter_side: str,
    priors: dict[str, pd.DataFrame],
    league_eng_sample: pd.DataFrame,
    pitcher_id: str | None,
) -> pd.DataFrame:
    """One synthetic row per pitch type the pitcher throws, using his observed medians."""
    d = pitcher_df.loc[_valid_pitch_mask(pitcher_df[pitch_col])].copy()
    if d.empty:
        return pd.DataFrame()
    rows = []
    for ptype, g in d.groupby(pitch_col, dropna=False):
        row: dict[str, Any] = {pitch_col: ptype, "prev_pitch_type": "__FIRST__"}
        for c in [
            "RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak",
            "PlateLocSide", "PlateLocHeight",
        ]:
            if c in g.columns:
                row[c] = pd.to_numeric(g[c], errors="coerce").median()
            else:
                row[c] = np.nan
        row["Balls"] = count_spec.get("balls", 0)
        row["Strikes"] = count_spec.get("strikes", 0)
        row["PitchofPA"] = count_spec.get("pitchofpa", 1)
        row["prev_RelSpeed"] = np.nan
        if "RelSpeed" in row and pd.notna(row["RelSpeed"]):
            row["delta_RelSpeed"] = 0.0
        else:
            row["delta_RelSpeed"] = np.nan
        zl, zr, zb, zt = -0.83, 0.83, 1.5, 3.5
        mid_h = (zb + zt) / 2.0
        mid_s = (zl + zr) / 2.0
        side = row.get("PlateLocSide", np.nan)
        height = row.get("PlateLocHeight", np.nan)
        if pd.notna(side) and pd.notna(height):
            row["in_zone"] = float(
                (side >= zl) and (side <= zr) and (height >= zb) and (height <= zt)
            )
            row["zone_upper"] = float(height > mid_h)
            bs = str(mode_batter_side).strip().lower()[:1]
            signed = float(side) if bs != "l" else -float(side)
            row["zone_inner"] = float(signed < mid_s)
        else:
            row["in_zone"] = np.nan
            row["zone_upper"] = np.nan
            row["zone_inner"] = np.nan
        st = float(row["Strikes"])
        b = float(row["Balls"])
        row["is_two_strike"] = float(st >= 2)
        row["is_hitter_count"] = float((b - st) >= 2 and b >= 2)
        row["is_pitcher_count"] = float(st >= 2 and b <= 1)

        pt = str(g["PitcherThrows"].mode().iloc[0]).strip() if "PitcherThrows" in g.columns and not g["PitcherThrows"].mode().empty else "Right"
        row["PitcherThrows"] = pt
        row["BatterSide"] = mode_batter_side
        row["platoon_same"] = float(_same_hand(pt, mode_batter_side))

        if "BatterId" in g.columns:
            bid = str(g["BatterId"].mode().iloc[0])
        else:
            bid = ""
        if "PitcherId" in g.columns:
            pid = str(g["PitcherId"].mode().iloc[0])
        else:
            pid = pitcher_id or ""

        tmp = pd.DataFrame([{**row, "BatterId": bid, "PitcherId": pid}])
        tmp = merge_priors(tmp, priors, pitch_col)
        rows.append(tmp.iloc[0].to_dict())

    return pd.DataFrame(rows)


def _predict_proba_three(models: dict[str, Any], X: pd.DataFrame, num_cols: list[str], cat_cols: list[str]):
    Xp = X[num_cols + cat_cols].copy()
    for c in cat_cols:
        Xp[c] = Xp[c].astype(str).fillna("missing")
    return (
        models["strike"].predict_proba(Xp)[:, 1],
        models["whiff"].predict_proba(Xp)[:, 1],
        models["hard"].predict_proba(Xp)[:, 1],
    )


def _shap_sentence(model, X_raw: pd.DataFrame, num_cols: list[str], cat_cols: list[str], feat_display: list[str]) -> str | None:
    """Optional TreeSHAP on the inner boosted trees; skipped if library/model shape unsupported."""
    try:
        import shap  # type: ignore

        cal_list = getattr(model, "calibrated_classifiers_", None)
        if not cal_list:
            return None
        fitted = cal_list[0]
        pipe = getattr(fitted, "estimator", fitted)
        if not hasattr(pipe, "named_steps"):
            return None
        prep = pipe.named_steps["prep"]
        clf = pipe.named_steps["clf"]
        Xt = prep.transform(X_raw)
        if hasattr(clf, "get_booster"):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[1]
            vals = np.abs(np.ravel(sv[0]))
        else:
            return None
        names = _get_feature_names(num_cols, cat_cols)
        if len(names) != len(vals):
            names = [f"f{i}" for i in range(len(vals))]
        order = np.argsort(-vals)[:4]
        bits = [names[i] for i in order if i < len(names)]
        return "SHAP drivers: " + ", ".join(bits[:3]) + "."
    except Exception:
        return None


def _clean_driver_name(name: str) -> str:
    nm = str(name or "")
    if "::" in nm:
        _, nm = nm.split("::", 1)
    mapping = {
        "RelSpeed": "velocity",
        "SpinRate": "spin rate",
        "SpinAxis": "spin axis",
        "InducedVertBreak": "induced vertical break",
        "HorzBreak": "horizontal break",
        "PlateLocSide": "horizontal location",
        "PlateLocHeight": "vertical location",
        "Balls": "ball count",
        "Strikes": "strike count",
        "PitchofPA": "pitch number in plate appearance",
        "prev_RelSpeed": "previous pitch velocity",
        "delta_RelSpeed": "velocity delta vs previous pitch",
        "in_zone": "in-zone tendency",
        "zone_upper": "upper-zone tendency",
        "zone_inner": "inner-half tendency",
        "is_two_strike": "two-strike context",
        "is_hitter_count": "hitter-count context",
        "is_pitcher_count": "pitcher-count context",
        "platoon_same": "same-handed matchup",
        "b_whiff_s": "batter whiff tendency vs pitch type",
        "b_hard_s": "batter hard-contact tendency vs pitch type",
        "b_chase_s": "batter chase tendency vs pitch type",
        "p_strike_s": "pitcher strike tendency for pitch type",
        "p_whiff_s": "pitcher whiff tendency for pitch type",
        "p_hard_s": "pitcher hard-contact tendency for pitch type",
        "p_usage_sm": "pitcher usage tendency",
        "TaggedPitchType": "pitch type",
        "prev_pitch_type": "previous pitch type",
        "PitcherThrows": "pitcher handedness",
        "BatterSide": "batter side",
    }
    return mapping.get(nm, nm.replace("_", " ").lower())


def _fallback_driver_lists(row: pd.Series, baseline: pd.Series, top_n: int = 2) -> tuple[list[str], list[str]]:
    """Fallback directional drivers when SHAP is unavailable."""
    candidates: list[tuple[str, float]] = []
    feature_map = {
        "RelSpeed": ("higher velocity", "lower velocity"),
        "InducedVertBreak": ("more vertical break", "less vertical break"),
        "HorzBreak": ("more horizontal movement", "less horizontal movement"),
        "SpinRate": ("higher spin rate", "lower spin rate"),
        "PlateLocHeight": ("higher location", "lower location"),
        "PlateLocSide": ("more arm-side location drift", "more glove-side location drift"),
        "in_zone": ("more in-zone tendency", "less in-zone tendency"),
        "zone_upper": ("more upper-zone tendency", "less upper-zone tendency"),
        "zone_inner": ("more inner-half tendency", "less inner-half tendency"),
        "p_strike_s": ("stronger strike profile", "weaker strike profile"),
        "p_whiff_s": ("stronger whiff profile", "weaker whiff profile"),
        "p_hard_s": ("more hard-contact profile", "less hard-contact profile"),
        "p_usage_sm": ("more used pitch type", "less used pitch type"),
    }
    for col, (pos_label, neg_label) in feature_map.items():
        if col not in row.index or col not in baseline.index:
            continue
        try:
            rv = float(row[col])
            bv = float(baseline[col])
        except Exception:
            continue
        if pd.isna(rv) or pd.isna(bv):
            continue
        diff = rv - bv
        if abs(diff) < 1e-6:
            continue
        candidates.append((pos_label if diff > 0 else neg_label, abs(diff)))
    candidates.sort(key=lambda x: -x[1])
    labels = [c[0] for c in candidates[: max(1, top_n * 2)]]
    pos = labels[:top_n]
    neg = labels[top_n : top_n * 2]
    if not neg:
        neg = ["limited counter-signal in profile"]
    return pos, neg


def _shap_driver_lists(model, X_raw: pd.DataFrame, num_cols: list[str], cat_cols: list[str], top_n: int = 2) -> tuple[list[str], list[str]]:
    """
    Return signed top drivers from TreeSHAP when supported.
    Positive drivers push the target up, negative drivers push the target down.
    """
    try:
        import shap  # type: ignore

        cal_list = getattr(model, "calibrated_classifiers_", None)
        if not cal_list:
            return [], []
        fitted = cal_list[0]
        pipe = getattr(fitted, "estimator", fitted)
        if not hasattr(pipe, "named_steps"):
            return [], []
        prep = pipe.named_steps["prep"]
        clf = pipe.named_steps["clf"]
        Xt = prep.transform(X_raw)
        if not hasattr(clf, "get_booster"):
            return [], []
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        vals = np.ravel(sv[0])
        names = _get_feature_names(num_cols, cat_cols)
        if len(names) != len(vals):
            names = [f"f{i}" for i in range(len(vals))]
        order_pos = np.argsort(-vals)
        order_neg = np.argsort(vals)
        pos = [_clean_driver_name(names[i]) for i in order_pos if vals[i] > 0][:top_n]
        neg = [_clean_driver_name(names[i]) for i in order_neg if vals[i] < 0][:top_n]
        return pos, neg
    except Exception:
        return [], []


def compute_scores(p_strike: np.ndarray, p_whiff: np.ndarray, p_hard: np.ndarray, two_strike_row: bool):
    cal = SCORE_WEIGHT_ATTACK_STRIKE_CAL * (0.65 - p_strike) ** 2
    attack = (
        SCORE_WEIGHT_ATTACK_STRIKE * p_strike
        - SCORE_WEIGHT_ATTACK_HARD_PENALTY * p_hard
        - cal
    )
    put = (
        SCORE_WEIGHT_PUT_WHIFF * p_whiff
        - SCORE_WEIGHT_PUT_HARD_PENALTY * p_hard
    )
    if two_strike_row:
        put += SCORE_WEIGHT_PUT_TWOSTRIKE_BOOST * p_whiff
    danger = SCORE_WEIGHT_DANGER_HARD * p_hard - SCORE_WEIGHT_DANGER_STRIKE * p_strike
    return attack, put, danger


def compute_ml_prediction_bundle(
    pitcher_df: pd.DataFrame,
    league_df: pd.DataFrame,
    descriptive_by_ptype: pd.DataFrame,
    batter_side_filter: str = "all",
    pitch_col: str = PITCH_TYPE_COL_DEFAULT,
    zone_bounds: tuple[float, float, float, float] = (-0.83, 0.83, 1.5, 3.5),
) -> dict[str, Any]:
    """
    Main entry: league-trained models, pitcher-specific typical rows, matchup-aware priors.
    descriptive_by_ptype: output of build_prediction_by_pitch_type from app (pitch samples).
    """
    empty: dict[str, Any] = {
        "use_ml": False,
        "message": "",
        "df": pd.DataFrame(),
        "summary": {"best_strike": None, "best_putaway": None, "caution": None},
        "training_note": "",
        "metrics_note": "",
    }

    try:
        from sklearn.compose import ColumnTransformer  # noqa: F401
    except ImportError:
        empty["message"] = "scikit-learn not available; install scikit-learn (and optionally xgboost) for ML predictions."
        return empty

    if descriptive_by_ptype is None or descriptive_by_ptype.empty:
        empty["message"] = "No descriptive pitch summary available."
        return empty

    if league_df is None or league_df.empty:
        empty["message"] = "No league Trackman data available for training."
        return empty

    league_side = filter_league_for_side(league_df, batter_side_filter)
    if league_side.empty:
        empty["message"] = "No league rows for selected batter-side filter."
        return empty

    league_eng = engineer_pitch_features(league_side, pitch_col, zone_bounds)
    if league_eng is None or len(league_eng) < ML_MIN_TRAIN_ROWS:
        empty["message"] = f"Not enough league pitches for ML training (need ≥{ML_MIN_TRAIN_ROWS})."
        return empty

    priors = compute_smoothed_priors(league_eng, pitch_col)
    league_eng = merge_priors(league_eng, priors, pitch_col)

    # Fingerprint engineered league rows so cache matches training subset.
    fp = _league_fingerprint(league_eng)
    bundle = load_bundle_from_disk(fp)
    if bundle is None or bundle.fingerprint != fp:
        bundle = train_or_load_bundle(league_eng, pitch_col)
    if bundle is None:
        empty["message"] = "Model training failed (check class balance / features)."
        return empty

    # Representative batter side for inference rows
    if batter_side_filter == "right":
        mode_bs = "Right"
    elif batter_side_filter == "left":
        mode_bs = "Left"
    else:
        if "BatterSide" in pitcher_df.columns:
            mode_bs = str(pitcher_df["BatterSide"].mode().iloc[0]) if not pitcher_df["BatterSide"].mode().empty else "Right"
        else:
            mode_bs = "Right"

    pid = None
    if pitcher_df is not None and not pitcher_df.empty and "PitcherId" in pitcher_df.columns:
        pid = str(pitcher_df["PitcherId"].iloc[0])

    X_neutral = _typical_pitch_profile_rows(
        pitcher_df, pitch_col, {"balls": 0, "strikes": 0, "pitchofpa": 1},
        mode_bs, priors, league_eng, pid,
    )
    X_put = _typical_pitch_profile_rows(
        pitcher_df, pitch_col, {"balls": 0, "strikes": 2, "pitchofpa": 3},
        mode_bs, priors, league_eng, pid,
    )
    if X_neutral.empty:
        empty["message"] = "Could not build per-pitch-type profile for this pitcher."
        return empty

    models = bundle.models
    ps_n, pw_n, ph_n = _predict_proba_three(models, X_neutral, bundle.num_cols, bundle.cat_cols)
    _, pw_t, ph_t = _predict_proba_three(models, X_put, bundle.num_cols, bundle.cat_cols)

    # Command/damage use neutral-count simulation; put-away uses two-strike row probabilities.
    attack_n, _, danger_n = compute_scores(ps_n, pw_n, ph_n, two_strike_row=False)
    put_sc = (
        SCORE_WEIGHT_PUT_WHIFF * pw_t
        - SCORE_WEIGHT_PUT_HARD_PENALTY * ph_t
        + SCORE_WEIGHT_PUT_TWOSTRIKE_BOOST * pw_t
    )

    counts = descriptive_by_ptype.set_index(pitch_col)["pitch_count"].to_dict()
    warns = descriptive_by_ptype.set_index(pitch_col)["sample_warning"].to_dict()

    ml_rows = []
    feat_names = _get_feature_names(bundle.num_cols, bundle.cat_cols)
    baseline = X_neutral.median(numeric_only=True)

    for i, ptype in enumerate(X_neutral[pitch_col].values):
        n = int(counts.get(ptype, 0))
        warn = str(warns.get(ptype, "")).strip()
        parts = [w for w in [warn] if w]
        if n < 20:
            parts.append(ML_LOW_SAMPLE_ML_WARN)
        stab = (
            "Very low stability" if n < 8 else "Low stability" if n < 20 else "Moderate" if n < 45 else "Stable"
        )
        row = X_neutral.iloc[[i]]
        row_put = X_put.iloc[[i]] if i < len(X_put) else row
        ps, pw, ph = float(ps_n[i]), float(pw_n[i]), float(ph_n[i])
        comp = float(attack_n[i])

        expl = _shap_sentence(models["strike"], row, bundle.num_cols, bundle.cat_cols, feat_names)
        if not expl:
            expl = fallback_explanation(row.iloc[0], baseline, feat_names, ps, ph, "attack")

        fb_pos, fb_neg = _fallback_driver_lists(row.iloc[0], baseline, top_n=2)

        strike_pos, strike_neg = _shap_driver_lists(models["strike"], row, bundle.num_cols, bundle.cat_cols, top_n=2)
        strike_src = "SHAP" if strike_pos or strike_neg else "Fallback profile drivers"
        if not (strike_pos or strike_neg):
            strike_pos, strike_neg = fb_pos, fb_neg

        put_pos, put_neg = _shap_driver_lists(models["whiff"], row_put, bundle.num_cols, bundle.cat_cols, top_n=2)
        put_src = "SHAP" if put_pos or put_neg else "Fallback profile drivers"
        if not (put_pos or put_neg):
            put_pos, put_neg = fb_pos, fb_neg

        hard_pos, hard_neg = _shap_driver_lists(models["hard"], row, bundle.num_cols, bundle.cat_cols, top_n=2)
        hard_src = "SHAP" if hard_pos or hard_neg else "Fallback profile drivers"
        if not (hard_pos or hard_neg):
            hard_pos, hard_neg = fb_pos, fb_neg

        ml_rows.append({
            pitch_col: ptype,
            "predicted_strike_prob": ps,
            "predicted_whiff_prob": pw,
            "predicted_hard_contact_prob": ph,
            "predicted_whiff_prob_putaway_context": float(pw_t[i]),
            "predicted_hard_contact_prob_putaway_context": float(ph_t[i]),
            "attack_score": float(attack_n[i]),
            "putaway_score": float(put_sc[i]),
            "danger_score": float(danger_n[i]),
            "composite_score": comp,
            "pitch_count": n,
            "sample_warning": " · ".join(parts) if parts else "",
            "stability_label": stab,
            "explanation_neutral": expl,
            "drivers_pos_attack": strike_pos,
            "drivers_neg_attack": strike_neg,
            "drivers_src_attack": strike_src,
            "drivers_pos_putaway": put_pos,
            "drivers_neg_putaway": put_neg,
            "drivers_src_putaway": put_src,
            "drivers_pos_caution": hard_pos,
            "drivers_neg_caution": hard_neg,
            "drivers_src_caution": hard_src,
        })

    ml_df = pd.DataFrame(ml_rows)

    # Recommendations from scores with sample guardrails
    def _pick_best(score_col: str, min_n: int, fallback_min: int, higher_better: bool = True):
        tier = ml_df[ml_df["pitch_count"] >= min_n]
        if tier.empty:
            tier = ml_df[ml_df["pitch_count"] >= fallback_min]
        if tier.empty:
            tier = ml_df
        if tier.empty:
            return None
        idx = tier[score_col].idxmax() if higher_better else tier[score_col].idxmin()
        return ml_df.loc[idx]

    best_attack = _pick_best("attack_score", PREDICTION_MIN_STRIKE_N, 6)
    best_put = _pick_best("putaway_score", PREDICTION_MIN_SWINGS_PUTAWAY, 4)
    best_danger = _pick_best("danger_score", PREDICTION_MIN_CONTACT_CAUTION, 3)

    mnote = _matchup_note(batter_side_filter)
    metrics_note = "; ".join(
        f"{k} ROC-AUC={v.get('roc_auc', 0):.3f}" for k, v in bundle.metrics.items()
    )
    bundle_warnings = getattr(bundle, "warnings", {}) or {}
    warning_note = " ".join(bundle_warnings.values()).strip()

    def _card_from_row(row: pd.Series | None, key: str, title_blurb: str, kind: str):
        if row is None:
            return None
        def _as_list(v):
            if isinstance(v, list):
                return v
            if isinstance(v, tuple):
                return list(v)
            return []
        p = str(row[pitch_col])
        expl = str(row.get("explanation_neutral", ""))
        if kind == "putaway":
            pr = X_put.loc[X_put[pitch_col].astype(str) == p]
            prow = pr.iloc[0] if not pr.empty else row
            expl = fallback_explanation(
                prow,
                baseline,
                feat_names,
                float(row["predicted_strike_prob"]),
                float(row["predicted_hard_contact_prob_putaway_context"]),
                "putaway",
            )
        if kind == "caution":
            expl = fallback_explanation(
                row,
                baseline,
                feat_names,
                float(row["predicted_strike_prob"]),
                float(row["predicted_hard_contact_prob"]),
                "caution",
            )
        return {
            "pitch": p,
            "pred_strike": float(row["predicted_strike_prob"]),
            "pred_whiff": float(row["predicted_whiff_prob_putaway_context"] if kind == "putaway" else row["predicted_whiff_prob"]),
            "pred_hard": float(row["predicted_hard_contact_prob_putaway_context"] if kind == "putaway" else row["predicted_hard_contact_prob"]),
            "pitch_count": int(row["pitch_count"]),
            "sample_warning": str(row["sample_warning"]),
            "coach_blurb": title_blurb,
            "is_ml": True,
            "explanation": expl,
            "matchup_note": mnote,
            "stability_label": str(row["stability_label"]),
            "composite": float(row["composite_score"]) if kind == "attack" else float(row["putaway_score"] if kind == "putaway" else row["danger_score"]),
            "drivers_pos": _as_list(row.get(f"drivers_pos_{kind}", [])),
            "drivers_neg": _as_list(row.get(f"drivers_neg_{kind}", [])),
            "drivers_source": str(row.get(f"drivers_src_{kind}", "Fallback profile drivers")),
        }

    summary = {
        "best_strike": _card_from_row(
            best_attack, "best_strike",
            "Best strike/command pitch for neutral counts in this matchup filter.",
            "attack",
        ),
        "best_putaway": _card_from_row(
            best_put, "best_putaway",
            "Best put-away option vs this batter-side pool (two-strike simulation).",
            "putaway",
        ),
        "caution": _card_from_row(
            best_danger, "caution",
            "Highest damage-risk pitch in this profile.",
            "caution",
        ),
    }

    # Labels for table
    def _rec_label(row):
        tags = []
        if best_attack is not None and row[pitch_col] == best_attack[pitch_col]:
            tags.append("Best command")
        if best_put is not None and row[pitch_col] == best_put[pitch_col]:
            tags.append("Best put-away")
        if best_danger is not None and row[pitch_col] == best_danger[pitch_col]:
            tags.append("Damage risk")
        return "; ".join(tags) if tags else "—"

    ml_df["recommendation_label"] = ml_df.apply(_rec_label, axis=1)

    return {
        "use_ml": True,
        "message": "",
        "df": ml_df,
        "summary": summary,
        "training_note": "Models trained on full league Trackman sample (gradient boosting + isotonic calibration). Pitch cards use this pitcher's typical shape per type.",
        "metrics_note": metrics_note,
        "warning_note": warning_note,
    }


def format_ml_prediction_table_display(pred: pd.DataFrame, summary: dict) -> pd.DataFrame:
    if pred is None or pred.empty:
        return pd.DataFrame(columns=[
            "Pitch", "Strike %", "Whiff %", "Hard Contact Risk", "Role", "Recommendation",
        ])
    pitch_col = PITCH_TYPE_COL_DEFAULT if PITCH_TYPE_COL_DEFAULT in pred.columns else pred.columns[0]
    rows = []
    for _, row in pred.iterrows():
        rec = str(row.get("recommendation_label", "—"))
        role = "Situational"
        if "Best command" in rec:
            role = "Command"
        elif "Best put-away" in rec:
            role = "Put-away"
        elif "Damage risk" in rec:
            role = "Risk"
        rows.append({
            "Pitch": str(row[pitch_col]),
            "Strike %": f"{float(row['predicted_strike_prob']) * 100:.1f}%",
            "Whiff %": f"{float(row['predicted_whiff_prob']) * 100:.1f}%",
            "Hard Contact Risk": f"{float(row['predicted_hard_contact_prob']) * 100:.1f}%",
            "Role": role,
            "Recommendation": rec,
        })
    return pd.DataFrame(rows)
