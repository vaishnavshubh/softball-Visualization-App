# Prediction Tab Logic and ML Layer

This document reflects the current Prediction tab implementation:

- pitcher-only UI recommendations
- descriptive fallback logic in `app.py`
- league-trained ML pipeline in `prediction_pipeline.py`
- score-based recommendation selection + model-driver explanations

## 1) Scope and Entry Points

Prediction is available only for **pitcher view** and **Trackman** data.

Core reactives/functions:

- `prediction_df()` -> `build_prediction_by_pitch_type(pitcher_data())`
- `prediction_summary()` -> `select_prediction_summary(prediction_df())` (descriptive fallback)
- `ml_prediction_state()` -> `prediction_pipeline.compute_ml_prediction_bundle(...)`
- `prediction_content()` decides ML cards vs historical fallback cards
- `prediction_table()` chooses `prediction_pipeline.format_ml_prediction_table_display(...)` when ML is active

High-level flow:

1. Build descriptive per-pitch-type stats for the selected pitcher.
2. Try league-trained ML scoring (filtered by current batter-side context).
3. If ML is available, render ML strategy cards + advanced impacts/drivers.
4. If ML is unavailable, render descriptive historical recommendations and fallback table.

---

## 2) Descriptive Layer (Fallback + Baseline Context)

Primary function: `build_prediction_by_pitch_type(df)` in `app.py`.

Per pitch type it computes:

- `pitch_count`
- `strike_pct`
- `contact_pct`
- `swing_count`
- `whiff_count`
- `contact_count`
- `whiff_pct` (`whiff_count / swing_count`)
- `hard_contact_count`
- `hard_contact_risk` (weighted score: `(0.5*firm_contact + hard_contact) / contact_count`)
- `sample_warning`

Event definitions:

- **Swing events**: `StrikeSwinging`, `FoulBallFieldable`, `FoulBallNotFieldable`, `InPlay`
- **Strike events**: `StrikeCalled`, `StrikeSwinging`, `FoulBallFieldable`, `FoulBallNotFieldable`, `InPlay`
- **Contact events**: `FoulBallFieldable`, `FoulBallNotFieldable`, `InPlay`
- EV interpretation bands:
  - typical D1 contact: `70-75 mph`
  - firm contact: `75-80 mph`
  - high-damage contact: `>=80 mph`

Descriptive recommendation selector (`select_prediction_summary(pred)`):

- **Best strike**: highest `strike_pct` with pitch-count tiers
- **Best put-away**: highest `whiff_pct` with swing-count tiers
- **Caution**: highest `hard_contact_risk` with contact-count tiers

Tiers:

- strike: `pitch_count >= 12`, fallback `>= 6`, then any
- put-away: `swing_count >= 8`, fallback `>= 4`, then any with swings
- caution: `contact_count >= 5`, fallback `>= 2`

---

## 3) ML Layer Overview (Current)

Primary function: `compute_ml_prediction_bundle(...)` in `prediction_pipeline.py`.

The ML pipeline now uses **league-trained gradient-boosted classifiers** (with calibrated probabilities) rather than the older pitcher-only logistic setup.

Three targets are modeled independently:

1. strike probability
2. whiff probability
3. hard-contact probability

Returned bundle keys:

- `use_ml`
- `message` (fallback reason when ML not used)
- `df` (per-pitch predictions + scores + explanations)
- `summary` (best command / best put-away / highest damage-risk cards)
- `training_note`
- `metrics_note`
- `warning_note`

---

## 4) Training Data, Context, and Gating

Training pool behavior:

- Models are trained from **league Trackman pool** (`league_df`), not just pitcher-only rows.
- League rows are filtered by `batter_side_filter` (`all` / `right` / `left`) before training/scoring.
- Pitch-type validity filtering removes blanks/undefined/other-like values.

Key gates:

- `ML_MIN_TRAIN_ROWS = 200` (league rows after engineering/filtering)
- `ML_MIN_PER_CLASS = 25` for non-fallback model fitting
- `ML_MAX_TRAIN_ROWS = 120000` (subsample cap for latency)

Fallback behavior during training:

- If class balance is too sparse or degenerate for a target, uses `ConstantProbModel` prior fallback for that target.
- `warning_note` surfaces these fallback-prior warnings in UI.

Dependency fallback:

- If `scikit-learn` is unavailable, ML is disabled and descriptive mode is used.

---

## 5) Feature Engineering and Priors

Feature engineering (`engineer_pitch_features`) includes:

- pitch shape/velo/location/count fields (`RelSpeed`, `SpinRate`, `SpinAxis`, `InducedVertBreak`, `HorzBreak`, `PlateLocSide`, `PlateLocHeight`, `Balls`, `Strikes`, `PitchofPA`)
- lag/context features (`prev_pitch_type`, `prev_RelSpeed`, `delta_RelSpeed`)
- zone/context flags (`in_zone`, `zone_upper`, `zone_inner`, `is_two_strike`, `is_hitter_count`, `is_pitcher_count`, `platoon_same`)

Smoothed priors (`compute_smoothed_priors` + `merge_priors`) add:

- batter x pitch-type smoothed tendencies (`b_whiff_s`, `b_hard_s`, `b_chase_s`)
- pitcher x pitch-type smoothed tendencies (`p_strike_s`, `p_whiff_s`, `p_hard_s`, `p_usage_sm`)
- empirical-Bayes-like shrinkage with `PRIOR_K = 25`

Target labels:

- `y_strike`: strike-event set
- `y_whiff`: `StrikeSwinging`
- `y_hard`: `InPlay` with EV `>= 80 mph`

---

## 6) Model Stack and Calibration

Estimator selection:

- prefer `xgboost.XGBClassifier` if available
- else `lightgbm.LGBMClassifier` if available
- else `sklearn.ensemble.HistGradientBoostingClassifier`

Pipeline:

- `ColumnTransformer`
  - numeric: median imputation
  - categorical: constant imputation + ordinal encoding with unknown handling
- probability calibration with `CalibratedClassifierCV(method="isotonic", cv="prefit")`

Validation metrics:

- stored per target (e.g., ROC-AUC, PR-AUC, log-loss when available)
- aggregated into `metrics_note` for Prediction tab display

Caching:

- in-memory cache + on-disk cache file `.prediction_model_cache.pkl`
- fingerprint-driven reuse from league date/min/max + row count signature
- UI button `Retrain Prediction Models` clears cache via `clear_prediction_cache(remove_disk=True)`

---

## 7) Inference Profiles and Decision Scores

Per pitch type, synthetic pitcher-specific profile rows are built from observed medians:

- neutral context row: `0-0`, early PA
- put-away context row: `0-2`, later PA

Predictions produced:

- `predicted_strike_prob`
- `predicted_whiff_prob`
- `predicted_hard_contact_prob`
- plus put-away-context versions for whiff/hard-contact

Decision scores:

- `attack_score` (command-oriented: strike up, hard-contact down, command calibration term)
- `putaway_score` (two-strike whiff reward minus hard-contact penalty + two-strike boost)
- `danger_score` (hard-contact risk emphasis with strike reliability adjustment)
- `composite_score` (overall profile impact used for ranking)

Stability / sample overlays:

- `pitch_count < 20` appends: `Low sample — estimates lean on league priors`
- stability labels: `Very low stability`, `Low stability`, `Moderate`, `Stable`

---

## 8) Recommendation Selection and Labels

ML recommendation picks are score-based with sample guardrails:

- **Best command**: max `attack_score` (`>=12`, fallback `>=6`)
- **Best put-away**: max `putaway_score` (`>=8`, fallback `>=4`)
- **Highest risk**: max `danger_score` (`>=5`, fallback `>=3`)

ML table recommendation tags:

- `Best command`
- `Best put-away`
- `Damage risk`

Fallback descriptive table tags (when ML disabled):

- `Best strike`
- `Best put-away`
- `Caution`

---

## 9) Explanations, Drivers, and Advanced UI

Card explanation text:

- built from SHAP when available (`_shap_sentence`, `_shap_driver_lists`)
- otherwise uses directional fallback profile deltas (`fallback_explanation`, `_fallback_driver_lists`)

Each ML summary card carries:

- predicted strike/whiff/hard probabilities (context-aware for put-away card)
- matchup note (RHB / LHB / combined)
- sample warning + stability label
- driver lists: upward/downward factors + source (`SHAP` or fallback drivers)

Advanced section (`prediction_content` / `prediction_advanced_*`):

- impact bars for control / two-strike put-away / hard-contact-risk effects
- model driver cards by recommendation type
- sortable advanced table with impact scores

---

## 10) Fallback and Messaging

If ML cannot run, fallback message in `message` may indicate:

- `scikit-learn` unavailable
- no league data under current filters
- insufficient league rows for ML training
- profile/scoring/training failure

When fallback happens:

- cards show historical-only framing
- table uses descriptive metrics
- small-sample warnings still appear from descriptive layer

---

## 11) Quick Function Reference

Descriptive layer (`app.py`):

- `build_prediction_by_pitch_type`
- `select_prediction_summary`
- `format_prediction_table_display`

ML layer (`prediction_pipeline.py`):

- `compute_ml_prediction_bundle`
- `engineer_pitch_features`
- `compute_smoothed_priors`
- `merge_priors`
- `train_or_load_bundle`
- `clear_prediction_cache`
- `format_ml_prediction_table_display`

