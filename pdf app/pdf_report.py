"""
Softball Pitch Dashboard (Uploaded CSV version)

This app mirrors the main dashboard visuals but uses a single user-uploaded
CSV that has the same structure/schema/column names as the existing TrackMan
CSVs (e.g., the ones read by app.py).
"""

import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.backends.backend_pdf import PdfPages
import tempfile

import numpy as np
import pandas as pd
from shiny import App, render, ui, reactive


# ---------------------------------------------------------------------------
# Config / constants (mirrors app.py where needed)
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")

PURDUE_LOGO_SRC = "PU-H-Full-Rev-RGB.png"

# Compact figure size for side-by-side web charts (PDF uses its own layout)
FIG_SIZE = (2.85, 2.15)
WEB_PIE_FIGSIZE = (3.35, 2.75)

COLUMNS_TO_KEEP = [
    "PitchNo", "Date", "Time", "PitchofPA",
    "Pitcher", "PitcherId", "PitcherThrows", "PitcherTeam",
    "Batter", "BatterId", "BatterSide", "BatterTeam",
    "Balls", "Strikes", "PitchCall", "TaggedPitchType",
    "RelSpeed", "SpinRate", "SpinAxis", "Tilt",
    "InducedVertBreak", "HorzBreak",
    "PlateLocHeight", "PlateLocSide",
    "TaggedHitType", "PlayResult", "ExitSpeed", "Angle", "Direction",
]

PITCH_TYPE_COL = "TaggedPitchType"
X_MOV = "HorzBreak"
Y_MOV = "InducedVertBreak"

# Fixed strike zone (feet)
ZONE_LEFT, ZONE_RIGHT = -0.83, 0.83
ZONE_BOTTOM, ZONE_TOP = 1.5, 3.5

MOV_XLIM = (-20, 20)
MOV_YLIM = (-20, 20)

# Purdue detection
PURDUE_CODE = "PUR_BOI_SB"


# ---------------------------------------------------------------------------
# Helpers (copied / adapted from app.py)
# ---------------------------------------------------------------------------
def is_purdue_team(team_value: str) -> bool:
    if team_value is None or pd.isna(team_value):
        return False
    return str(team_value).strip() == PURDUE_CODE


def _has_any_numeric(series: pd.Series) -> bool:
    if series is None or series.empty:
        return False
    x = pd.to_numeric(series, errors="coerce")
    return x.notna().any()


def _has_any_text(series: pd.Series) -> bool:
    if series is None or series.empty:
        return False
    x = series.astype("string").str.strip().dropna()
    x = x[x.str.lower() != "nan"]
    return (x != "").any()


def infer_session_type_for_purdue(df: pd.DataFrame) -> str:
    """
    Purdue-only rule per file:
      Bullpen: Purdue pitching exists, Purdue batted-ball does not
      Batting Practice: Purdue batted-ball exists, Purdue pitching does not
      Scrimmage: both exist
    """
    if df is None or df.empty:
        return "Unknown"

    pur_pitch = df[df.get("PitcherTeam", "").astype("string").str.strip() == PURDUE_CODE] if "PitcherTeam" in df.columns else df.iloc[0:0]
    pur_bat = df[df.get("BatterTeam", "").astype("string").str.strip() == PURDUE_CODE] if "BatterTeam" in df.columns else df.iloc[0:0]

    has_pitch = False
    if not pur_pitch.empty:
        pitch_signals = []
        if "TaggedPitchType" in pur_pitch.columns:
            pitch_signals.append(_has_any_text(pur_pitch["TaggedPitchType"]))
        if "RelSpeed" in pur_pitch.columns:
            pitch_signals.append(_has_any_numeric(pur_pitch["RelSpeed"]))
        if "PlateLocSide" in pur_pitch.columns:
            pitch_signals.append(_has_any_numeric(pur_pitch["PlateLocSide"]))
        if "PlateLocHeight" in pur_pitch.columns:
            pitch_signals.append(_has_any_numeric(pur_pitch["PlateLocHeight"]))
        has_pitch = any(pitch_signals) if pitch_signals else True

    has_bat = False
    if not pur_bat.empty:
        bat_signals = []
        if "ExitSpeed" in pur_bat.columns:
            bat_signals.append(_has_any_numeric(pur_bat["ExitSpeed"]))
        if "Angle" in pur_bat.columns:
            bat_signals.append(_has_any_numeric(pur_bat["Angle"]))
        if "Direction" in pur_bat.columns:
            bat_signals.append(_has_any_numeric(pur_bat["Direction"]))
        if "TaggedHitType" in pur_bat.columns:
            bat_signals.append(_has_any_text(pur_bat["TaggedHitType"]))
        if "PlayResult" in pur_bat.columns:
            bat_signals.append(_has_any_text(pur_bat["PlayResult"]))
        has_bat = any(bat_signals) if bat_signals else True

    if has_pitch and has_bat:
        return "Scrimmage"
    if has_pitch and not has_bat:
        return "Bullpen"
    if has_bat and not has_pitch:
        return "Batting Practice"
    return "Unknown"


def apply_session_filter_for_team(df: pd.DataFrame, team: str, session_key: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not is_purdue_team(team):
        return df

    if session_key == "bullpen":
        return df[df["SessionType"] == "Bullpen"]
    if session_key == "bp":
        return df[df["SessionType"] == "Batting Practice"]
    if session_key == "scrimmage":
        return df[df["SessionType"] == "Scrimmage"]
    return df


PITCH_TYPE_FIXED_COLORS = {
    "Fastball":  "#2563EB",
    "Changeup":  "#10B981",
    "Curveball": "#8B5CF6",
    "Riseball":  "#0EA5E9",
    "Dropball":  "#EF4444",
    "Screwball": "#EC4899",
    "Offspeed":  "#F59E0B",
    "Riser": "#0EA5E9",
    "Rise":  "#0EA5E9",
    "Drop":  "#EF4444",
}
PITCH_TYPE_FALLBACK_COLORS = ["#06B6D4", "#F97316", "#84CC16", "#6366F1"]


def build_pitch_color_map(pitch_types):
    out = {}
    unknown = sorted([p for p in pitch_types if pd.notna(p) and p not in PITCH_TYPE_FIXED_COLORS])
    for pt in pitch_types:
        if pd.isna(pt):
            continue
        if pt in PITCH_TYPE_FIXED_COLORS:
            out[pt] = PITCH_TYPE_FIXED_COLORS[pt]
        else:
            idx = unknown.index(pt) if pt in unknown else 0
            out[pt] = PITCH_TYPE_FALLBACK_COLORS[idx % len(PITCH_TYPE_FALLBACK_COLORS)]
    return out


def home_plate_polygon(y_front=0.0):
    half_width = (17 / 12) / 2
    diag = 8.5 / 12
    back = 12 / 12
    pts = np.array([
        [-half_width, y_front],
        [half_width, y_front],
        [half_width - diag, y_front - diag],
        [0, y_front - back],
        [-(half_width - diag), y_front - diag],
    ])
    return Polygon(pts, closed=True, fill=False, linewidth=2)


def format_display_name(raw_name: str) -> str:
    if not raw_name or pd.isna(raw_name):
        return ""
    s = str(raw_name).strip()
    if "," in s:
        last, first = s.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return s


def compute_usage(df: pd.DataFrame, pitcher_id) -> pd.DataFrame:
    if df is None or df.empty or pitcher_id is None:
        return pd.DataFrame(columns=[PITCH_TYPE_COL, "pitch_count", "usage_pct"])
    d = df[(df["PitcherId"].astype(str) == str(pitcher_id)) & df[PITCH_TYPE_COL].notna()]
    if d.empty:
        return pd.DataFrame(columns=[PITCH_TYPE_COL, "pitch_count", "usage_pct"])
    usage = (
        d[PITCH_TYPE_COL]
        .value_counts()
        .rename_axis(PITCH_TYPE_COL)
        .reset_index(name="pitch_count")
    )
    usage["usage_pct"] = usage["pitch_count"] / usage["pitch_count"].sum()
    return usage.sort_values("usage_pct", ascending=False).reset_index(drop=True)


def compute_pitch_metrics(df: pd.DataFrame, pitcher_id):
    if df is None or df.empty or pitcher_id is None:
        return pd.DataFrame()

    required = {
        "PitcherId", PITCH_TYPE_COL, "PitchNo", "RelSpeed",
        "PitchCall", "PlateLocSide", "PlateLocHeight"
    }
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    d = df[
        (df["PitcherId"].astype(str) == str(pitcher_id)) &
        df[PITCH_TYPE_COL].notna()
    ].copy()

    if d.empty:
        return pd.DataFrame()

    d["RelSpeed"] = pd.to_numeric(d["RelSpeed"], errors="coerce")
    d["PlateLocSide"] = pd.to_numeric(d["PlateLocSide"], errors="coerce")
    d["PlateLocHeight"] = pd.to_numeric(d["PlateLocHeight"], errors="coerce")

    if "SpinRate" in d.columns:
        d["SpinRate"] = pd.to_numeric(d["SpinRate"], errors="coerce")
    else:
        d["SpinRate"] = np.nan

    pc = d["PitchCall"].astype(str).str.strip()

    STRIKE_EVENTS = {
        "StrikeCalled", "StrikeSwinging",
        "FoulBallFieldable", "FoulBallNotFieldable",
        "InPlay"
    }
    SWING_EVENTS = {
        "StrikeSwinging",
        "FoulBallFieldable", "FoulBallNotFieldable",
        "InPlay"
    }
    CONTACT_EVENTS = {
        "FoulBallFieldable", "FoulBallNotFieldable",
        "InPlay"
    }

    d["is_strike"] = pc.isin(STRIKE_EVENTS)
    d["is_called_strike"] = pc.eq("StrikeCalled")
    d["is_swing"] = pc.isin(SWING_EVENTS)
    d["is_whiff"] = pc.eq("StrikeSwinging")
    d["is_contact"] = pc.isin(CONTACT_EVENTS)

    loc_valid = d["PlateLocSide"].notna() & d["PlateLocHeight"].notna()

    d["is_in_zone"] = (
        loc_valid &
        d["PlateLocSide"].between(ZONE_LEFT, ZONE_RIGHT, inclusive="both") &
        d["PlateLocHeight"].between(ZONE_BOTTOM, ZONE_TOP, inclusive="both")
    )
    d["is_outside"] = loc_valid & ~d["is_in_zone"]

    d["is_zone_swing"] = d["is_in_zone"] & d["is_swing"]
    d["is_zone_contact"] = d["is_in_zone"] & d["is_contact"]

    d["is_chase"] = d["is_outside"] & d["is_swing"]
    d["is_chase_contact"] = d["is_outside"] & d["is_contact"]

    summary = (
        d.groupby(PITCH_TYPE_COL, dropna=True)
        .agg(
            pitch_count=("PitchNo", "count"),
            max_velo=("RelSpeed", "max"),
            avg_velo=("RelSpeed", "mean"),
            spin_rate=("SpinRate", "mean"),
            strike_pct=("is_strike", "mean"),
            called_strike_n=("is_called_strike", "sum"),
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum"),
            contacts=("is_contact", "sum"),
            in_zone_n=("is_in_zone", "sum"),
            zone_swings=("is_zone_swing", "sum"),
            zone_contacts=("is_zone_contact", "sum"),
            outside_n=("is_outside", "sum"),
            chases=("is_chase", "sum"),
            chase_contacts=("is_chase_contact", "sum"),
        )
        .reset_index()
    )

    total = summary["pitch_count"].sum()
    summary["usage_pct"] = summary["pitch_count"] / total if total else 0

    summary["called_strike_pct"] = np.where(
        summary["pitch_count"] > 0,
        summary["called_strike_n"] / summary["pitch_count"],
        np.nan
    )
    summary["swing_pct"] = np.where(
        summary["pitch_count"] > 0,
        summary["swings"] / summary["pitch_count"],
        np.nan
    )
    summary["swstr_pct"] = np.where(
        summary["pitch_count"] > 0,
        summary["whiffs"] / summary["pitch_count"],
        np.nan
    )
    summary["whiff_pct"] = np.where(
        summary["swings"] > 0,
        summary["whiffs"] / summary["swings"],
        np.nan
    )
    summary["zone_swing_pct"] = np.where(
        summary["in_zone_n"] > 0,
        summary["zone_swings"] / summary["in_zone_n"],
        np.nan
    )
    summary["zone_contact_pct"] = np.where(
        summary["zone_swings"] > 0,
        summary["zone_contacts"] / summary["zone_swings"],
        np.nan
    )
    summary["chase_pct"] = np.where(
        summary["outside_n"] > 0,
        summary["chases"] / summary["outside_n"],
        np.nan
    )
    summary["chase_contact_pct"] = np.where(
        summary["chases"] > 0,
        summary["chase_contacts"] / summary["chases"],
        np.nan
    )

    return (
        summary.drop(
            columns=[
                "called_strike_n", "swings", "whiffs", "contacts",
                "in_zone_n", "zone_swings", "zone_contacts",
                "outside_n", "chases", "chase_contacts"
            ]
        )
        .sort_values("usage_pct", ascending=False)
        .reset_index(drop=True)
    )


def throws_to_short(throws: str) -> str:
    if not throws:
        return ""
    t = str(throws).strip().lower()
    if "right" in t:
        return "RHP"
    if "left" in t:
        return "LHP"
    return str(throws).strip()


def format_num(x, digits=1):
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def format_pct(x, digits=1):
    if pd.isna(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.tags.style("""
        body {
            background-color: #f3f3f3;
            margin: 0;
            font-family: Arial, sans-serif;
        }

        .sidebar .shiny-input-container,
        .sidebar .filter-title {
            font-weight: 900 !important;
        }

        .sidebar .shiny-input-container .form-check label.form-check-label {
            font-weight: 400 !important;
        }

        /* Purdue header */
        .top-header {
            background-color: #000000;
            border-bottom: 6px solid #DDB945;
            height: 64px;
            display: flex;
            align-items: center;
            padding: 0 18px;
            box-sizing: border-box;
        }
        .top-header .logo-wrap {
            width: 160px;
            display: flex;
            align-items: center;
        }
        .top-header .logo-wrap img {
            height: 40px;
            width: auto;
            display: block;
        }
        .top-header .title {
            flex: 1;
            text-align: center;
            color: #DDB945;
            font-size: 40px;
            font-weight: 900;
            letter-spacing: 0.5px;
        }
        .top-header .spacer {
            width: 160px;
        }

        .layout-main {
            display: flex;
            width: 100%;
            min-height: calc(100vh - 64px);
        }

        .sidebar {
            width: 260px;
            background-color: #ffffff;
            border-right: 1px solid #d6d6d6;
            padding: 16px 16px 20px 16px;
            box-sizing: border-box;
        }

        .sidebar h4 {
            margin: 0 0 12px 0;
            font-size: 24px;
            font-weight: 900;
            text-align: center;
        }

        .main-area {
            flex: 1;
            padding: 16px 20px 22px 20px;
            box-sizing: border-box;
        }

        .panel {
            background: #ffffff;
            border: 1px solid #d6d6d6;
            border-radius: 10px;
            padding: 16px 16px 18px 16px;
            min-height: calc(100vh - 140px);
        }

        .profile-title {
            font-size: 24px;
            font-weight: 900;
            margin: 0 0 10px 0;
            text-align: center;
        }

        .player-summary {
            font-size: 20px;
            font-weight: 800;
            color: #333333;
            margin: 0 0 14px 0;
            text-align: center;
        }

        .grid-2x2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        .card {
            background: #f7f7f7;
            border: 1px solid #d6d6d6;
            border-radius: 10px;
            padding: 10px;
        }

        .legend-row {
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            justify-content: center;
            align-items: center;
            margin: 8px 0 14px 0;
            padding: 8px 14px;
            border: 1px solid rgba(0,0,0,0.15);
            border-radius: 8px;
            background: #fff;
            font-size: 13px;
        }

        /* Usage table styling */
        .usage-table-wrap table,
        .usage-table-wrap thead,
        .usage-table-wrap tbody,
        .usage-table-wrap tr,
        .usage-table-wrap th,
        .usage-table-wrap td,
        .usage-table-wrap table.table,
        .usage-table-wrap .table,
        .usage-table-wrap .table > :not(caption) > * > * {
            background-color: transparent !important;
        }

        .usage-table-wrap table {
            width: auto !important;
            table-layout: fixed !important;
            border-collapse: collapse !important;
        }

        /* Header */
        .usage-table-wrap table.table thead th,
        .usage-table-wrap .table thead th {
            background: #111 !important;
            color: #fff !important;
            font-weight: 800 !important;
            border: 1px solid #444 !important;
            border-bottom: 2px solid #444 !important;
            opacity: 1 !important;
        }

        /* Body cells grid */
        .usage-table-wrap tbody td {
            border: 1px solid #cfcfcf !important;
            color: #111 !important;
        }

        /* Zebra striping */
        .usage-table-wrap tbody tr:nth-child(odd) td {
            background: #f3f3f3 !important;
        }

        .usage-table-wrap tbody tr:nth-child(even) td {
            background: #ffffff !important;
        }

        .usage-table-wrap tbody tr:hover td {
            background: #e9e9e9 !important;
        }

        .usage-table-wrap th,
        .usage-table-wrap td {
            padding: 3px 5px;
            font-size: 11px;
            line-height: 1.2;
            vertical-align: middle;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .usage-table-wrap tbody tr {
            height: 25px;
        }

        .usage-table-wrap th:nth-child(1),
        .usage-table-wrap td:nth-child(1) { text-align: left;  width: 160px; padding-right: 4px; }
        .usage-table-wrap th:nth-child(2),
        .usage-table-wrap td:nth-child(2) { text-align: right; width: 70px; padding-left: 4px; }
        .usage-table-wrap th:nth-child(3),
        .usage-table-wrap td:nth-child(3) { text-align: right; width: 85px; }
        .usage-table-wrap th:nth-child(4),
        .usage-table-wrap td:nth-child(4) { text-align: right; width: 85px; }
        .usage-table-wrap th:nth-child(5),
        .usage-table-wrap td:nth-child(5) { text-align: right; width: 85px; }
        .usage-table-wrap th:nth-child(6),
        .usage-table-wrap td:nth-child(6) { text-align: right; width: 95px; }
        .usage-table-wrap th:nth-child(7),
        .usage-table-wrap td:nth-child(7) { text-align: right; width: 70px; }
        .usage-table-wrap th:nth-child(8),
        .usage-table-wrap td:nth-child(8) { text-align: right; width: 70px; }
        .usage-table-wrap th:nth-child(9),
        .usage-table-wrap td:nth-child(9) { text-align: right; width: 75px; }

        .table-title {
            font-size: 16px;
            font-weight: 900;
            margin: 6px 0 10px 0;
            text-align: center;
        }
    """),

    # Purdue header (logo left, title centered)
    ui.tags.div(
        ui.tags.div(
            ui.tags.img(src=PURDUE_LOGO_SRC, alt="Purdue Logo"),
            class_="logo-wrap",
        ),
        ui.tags.div("Softball Dashboard", class_="title"),
        ui.tags.div(class_="spacer"),
        class_="top-header",
    ),

    ui.tags.div(
        ui.tags.div(
            ui.tags.h4("Filters"),
            ui.input_file(
                "file",
                "Upload CSV (TrackMan schema)",
                multiple=False,
                accept=[".csv"],
            ),
            ui.input_select(
                "session_type",
                "Session Type",
                choices={
                    "all": "All",
                    "bullpen": "Bullpen",
                    "bp": "Batting Practice",
                    "scrimmage": "Scrimmage",
                    "live": "Live Game",
                },
                selected="all",
            ),
            ui.input_select("player", "Player Name", choices={"": "—"}),
            ui.input_select(
                "batter_handedness",
                "Batter Handedness",
                choices={"all": "Combined View", "Right": "Right Handed", "Left": "Left Handed"},
                selected="all",
            ),
            ui.download_button("download_pdf", "Download PDF Report"),
            class_="sidebar",
        ),

        ui.tags.div(
            ui.output_ui("home_content"),
            class_="main-area",
        ),

        class_="layout-main",
    ),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):

    # ---- 1. Load uploaded CSV ----
    @reactive.calc
    def uploaded_df():
        f = input.file()
        if f is None or len(f) == 0:
            return None

        path = f[0]["datapath"]
        try:
            raw = pd.read_csv(path)
        except Exception:
            return None

        cols = [c for c in COLUMNS_TO_KEEP if c in raw.columns]
        if not cols:
            return None

        df = raw[cols].copy()
        for c in [
            "Pitcher", "Batter", PITCH_TYPE_COL, "PitchCall",
            "PitcherTeam", "BatterTeam", "PitcherThrows", "BatterSide"
        ]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().replace("nan", np.nan)

        # Normalize PitcherId/BatterId to string so dropdown and filters work (handles int/float/NaN)
        for c in ["PitcherId", "BatterId"]:
            if c in df.columns:
                s = df[c].astype(str).str.strip()
                s = s.replace("nan", "").replace("NaN", "")
                df[c] = s

        df["SessionType"] = infer_session_type_for_purdue(df)
        return df

    @reactive.calc
    def current_df():
        df = uploaded_df()
        if df is None:
            return None
        return apply_session_filter_for_team(df, PURDUE_CODE, input.session_type())

    @reactive.effect
    def _update_player_choices():
        df = current_df()
        if df is None or df.empty:
            ui.update_select("player", choices={"": "—"}, label="Player Name", session=session)
            return
        if "PitcherId" not in df.columns or "Pitcher" not in df.columns:
            ui.update_select("player", choices={"": "—"}, label="Player Name", session=session)
            return

        lookup = df[["PitcherId", "Pitcher"]].copy()
        lookup["PitcherId"] = lookup["PitcherId"].astype(str).str.strip()
        lookup = lookup[
            lookup["PitcherId"].str.len().gt(0)
            & lookup["PitcherId"].str.lower().ne("nan")
        ]
        lookup = lookup.drop_duplicates().sort_values(["Pitcher", "PitcherId"])

        choices = {
            str(r.PitcherId): (format_display_name(r.Pitcher) or str(r.PitcherId))
            for r in lookup.itertuples(index=False)
        }
        ui.update_select(
            "player",
            choices=choices,
            label="Player Name",
            selected=list(choices.keys())[0] if choices else None,
            session=session,
        )

    # ---- 4. Core reactive data for visuals ----
    @reactive.calc
    def pitcher_data():
        df = current_df()
        pid = input.player()
        if df is None or df.empty or not pid:
            return None
        d = df[df["PitcherId"].astype(str) == str(pid)].copy()
        b_side = input.batter_handedness()
        if b_side != "all" and "BatterSide" in d.columns:
            d = d[d["BatterSide"].astype(str).str.strip() == b_side]
        return d

    @reactive.calc
    def pitch_colors():
        data = pitcher_data()
        if data is None or PITCH_TYPE_COL not in data.columns:
            return {}
        return build_pitch_color_map(data[PITCH_TYPE_COL].dropna().unique())

    @reactive.calc
    def usage_df():
        data = pitcher_data()
        if data is None or data.empty:
            return pd.DataFrame(columns=[PITCH_TYPE_COL, "pitch_count", "usage_pct"])
        usage = (
            data[data[PITCH_TYPE_COL].notna()][PITCH_TYPE_COL]
            .value_counts()
            .rename_axis(PITCH_TYPE_COL)
            .reset_index(name="pitch_count")
        )
        if usage.empty:
            return pd.DataFrame(columns=[PITCH_TYPE_COL, "pitch_count", "usage_pct"])
        usage["usage_pct"] = usage["pitch_count"] / usage["pitch_count"].sum()
        return usage

    @reactive.calc
    def player_summary_text():
        data = pitcher_data()
        if data is None or data.empty:
            return "Select player to view profile"

        name_raw = data["Pitcher"].iloc[0] if "Pitcher" in data.columns else ""
        throws_raw = data["PitcherThrows"].iloc[0] if "PitcherThrows" in data.columns else ""
        name = format_display_name(name_raw) or "Pitcher"
        hand = throws_to_short(throws_raw)

        n_pitches = len(data)

        b_hand = input.batter_handedness()
        vs_str = ""
        if b_hand == "Right":
            vs_str = " (vs RHB)"
        elif b_hand == "Left":
            vs_str = " (vs LHB)"

        if hand:
            return f"{name} | {hand} {n_pitches} pitches{vs_str}"
        return f"{name} {n_pitches} pitches{vs_str}"

    @output
    @render.text
    def player_summary():
        return player_summary_text()

    @output
    @render.ui
    def home_content():
        df = current_df()
        if df is None or df.empty:
            return ui.div(
                "Upload a CSV (with the standard schema) and select a player.",
                class_="panel",
            )

        data = pitcher_data()
        if data is None or data.empty:
            return ui.div(
                ui.div("Pitch Profile", class_="profile-title"),
                ui.div("No pitcher data for the selected filters.", class_="player-summary"),
                class_="panel",
            )

        return ui.div(
            ui.div("Pitch Profile", class_="profile-title"),
            ui.div(ui.output_text("player_summary"), class_="player-summary"),
            ui.output_ui("movement_legend"),
            ui.row(
                ui.column(
                    4,
                    ui.card(
                        ui.card_header("Pitch Usage"),
                        ui.output_plot("pie", height="260px"),
                    ),
                ),
                ui.column(
                    4,
                    ui.card(
                        ui.card_header("Pitch Locations"),
                        ui.output_plot("location", height="260px"),
                    ),
                ),
                ui.column(
                    4,
                    ui.card(
                        ui.card_header("Pitch Movements"),
                        ui.output_plot("movement", height="260px"),
                    ),
                ),
            ),
            ui.row(
                ui.column(
                    12,
                    ui.card(
                        ui.card_header("Summary Table"),
                        ui.div(
                            ui.output_table("usage_table"),
                            class_="usage-table-wrap",
                        ),
                    ),
                ),
            ),
            class_="panel",
        )

    # ---- 6. Plots & table ----
    @output
    @render.plot
    def pie():
        usage = usage_df()
        colors = pitch_colors()

        fig, ax = plt.subplots(figsize=WEB_PIE_FIGSIZE)
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#f7f7f7")

        if usage.empty:
            ax.text(0.5, 0.5, "No pitch usage data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        labels = usage[PITCH_TYPE_COL].tolist()
        pcts = usage["usage_pct"].values
        c = [colors.get(pt, (0.5, 0.5, 0.5)) for pt in labels]

        ax.pie(
            pcts,
            labels=None,
            colors=c,
            startangle=90,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
            pctdistance=0.65,
            textprops={"fontsize": 8, "fontweight": "bold"},
        )
        for t in ax.texts:
            t.set_fontsize(7)

        ax.set_title("Pitch Usage", fontsize=12, fontweight="bold")
        return fig

    @output
    @render.plot
    def location():
        data = pitcher_data()
        colors = pitch_colors()

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#f7f7f7")

        if data is None or data.empty:
            ax.text(0.5, 0.5, "No pitch location data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        loc = data[
            data["PlateLocSide"].notna()
            & data["PlateLocHeight"].notna()
            & data[PITCH_TYPE_COL].notna()
        ]
        if loc.empty:
            ax.text(0.5, 0.5, "No pitch location data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        ax.add_patch(Rectangle(
            (ZONE_LEFT - 0.3, ZONE_BOTTOM - 0.3),
            (ZONE_RIGHT - ZONE_LEFT) + 0.6,
            (ZONE_TOP - ZONE_BOTTOM) + 0.6,
            facecolor="#d9d9d9",
            edgecolor="none",
            alpha=0.25
        ))
        ax.add_patch(Rectangle(
            (ZONE_LEFT, ZONE_BOTTOM),
            ZONE_RIGHT - ZONE_LEFT,
            ZONE_TOP - ZONE_BOTTOM,
            fill=False,
            linewidth=2
        ))
        ax.plot([ZONE_LEFT, ZONE_RIGHT], [(ZONE_BOTTOM + ZONE_TOP) / 2] * 2, linestyle="--", linewidth=1, color="#1f77b4")
        ax.plot([0, 0], [ZONE_BOTTOM, ZONE_TOP], linestyle="--", linewidth=1, color="#ff7f0e")

        for pt, g in loc.groupby(PITCH_TYPE_COL):
            ax.scatter(
                g["PlateLocSide"],
                g["PlateLocHeight"],
                s=22,
                alpha=0.8,
                color=colors.get(pt, (0.5, 0.5, 0.5)),
            )

        ax.add_patch(home_plate_polygon(y_front=0.10))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PlateLocSide", fontsize=8)
        ax.set_ylabel("PlateLocHeight", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_title("Pitch Locations", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)
        return fig

    @output
    @render.plot
    def movement():
        data = pitcher_data()
        colors = pitch_colors()

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#f7f7f7")

        if data is None or data.empty:
            ax.text(0.5, 0.5, "No pitch movement data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        mov = data[data[X_MOV].notna() & data[Y_MOV].notna() & data[PITCH_TYPE_COL].notna()]
        if mov.empty:
            ax.text(0.5, 0.5, "No pitch movement data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        for pt, g in mov.groupby(PITCH_TYPE_COL):
            ax.scatter(
                g[X_MOV], g[Y_MOV],
                s=16, alpha=0.75,
                color=colors.get(pt, (0.5, 0.5, 0.5))
            )

        ax.axhline(0, linewidth=1, color="#777777")
        ax.axvline(0, linewidth=1, color="#777777")
        ax.set_xlim(*MOV_XLIM)
        ax.set_ylim(*MOV_YLIM)
        ax.set_xlabel("Horizontal break (in)", fontsize=8)
        ax.set_ylabel("Induced vertical break (in)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_title("Pitch Movements", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)

        # Stats box
        lines = []
        for pt, g in mov.groupby(PITCH_TYPE_COL):
            avg_hb  = g[X_MOV].mean()
            avg_ivb = g[Y_MOV].mean()
            color   = colors.get(pt, "#555555")
            lines.append((pt, avg_hb, avg_ivb, color))

        box_text = "\n".join(
            f"{pt}: Avg HB {avg_hb:+.1f}, Avg IVB {avg_ivb:+.1f}"
            for pt, avg_hb, avg_ivb, _ in lines
        )
        ax.text(
            0.98, 0.98, box_text,
            transform=ax.transAxes,
            fontsize=5.5,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.85),
            family="monospace",
        )

        return fig

    @output
    @render.ui
    def movement_legend():
        df = pitcher_data()
        if df is None or df.empty or PITCH_TYPE_COL not in df.columns:
            return ui.div()

        order = (
            df[PITCH_TYPE_COL]
            .dropna()
            .astype(str)
            .value_counts()
            .index.tolist()
        )

        cols = pitch_colors()

        items = []
        for pt in order:
            if pt.lower() == "other":
                continue

            color = cols.get(pt, "#777777")
            items.append(
                ui.div(
                    ui.span(
                        style=f"display:inline-block; width:10px; height:10px; border-radius:50%; background:{color};"
                    ),
                    ui.span(pt),
                    style="display:flex; align-items:center; gap:6px;",
                )
            )

        return ui.div(
            *items,
            class_="legend-row",
        )

    @output
    @render.table
    def usage_table():
        data = pitcher_data()
        session_type = "live"
        if data is not None and not data.empty and "SessionType" in data.columns:
            st = str(data["SessionType"].iloc[0]).strip().lower()
            if st == "bullpen":
                session_type = "bullpen"
            elif st in {"scrimmage", "live", "batting practice"}:
                session_type = "live"

        bullpen_cols = [
            "Pitch type", "Count", "Usage %",
            "Max Velo", "Avg Velo", "Spin Rate", "Strike %"
        ]
        live_scrimmage_cols = [
            "Pitch type", "Count", "Usage %",
            "Max Velo", "Avg Velo", "Spin Rate",
            "Strike %", "Called Strike %",
            "Swing %", "SwStrike %",
            "Whiff %", "Zone Swing %",
            "Zone Contact %", "Chase %",
            "Chase Contact %"
        ]

        if data is None or data.empty:
            return pd.DataFrame(columns=bullpen_cols if session_type == "bullpen" else live_scrimmage_cols)

        # Build summary from selected player's already-filtered rows.
        dsum = data[data[PITCH_TYPE_COL].notna()].copy()
        if dsum.empty:
            return pd.DataFrame(columns=bullpen_cols if session_type == "bullpen" else live_scrimmage_cols)

        dsum["RelSpeed"] = pd.to_numeric(dsum["RelSpeed"], errors="coerce")
        dsum["PlateLocSide"] = pd.to_numeric(dsum["PlateLocSide"], errors="coerce")
        dsum["PlateLocHeight"] = pd.to_numeric(dsum["PlateLocHeight"], errors="coerce")
        dsum["SpinRate"] = pd.to_numeric(dsum["SpinRate"], errors="coerce") if "SpinRate" in dsum.columns else np.nan
        pc_sum = dsum["PitchCall"].astype(str).str.strip()

        strike_events = {"StrikeCalled", "StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        swing_events = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        contact_events = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}

        dsum["is_strike"] = pc_sum.isin(strike_events)
        dsum["is_called_strike"] = pc_sum.eq("StrikeCalled")
        dsum["is_swing"] = pc_sum.isin(swing_events)
        dsum["is_whiff"] = pc_sum.eq("StrikeSwinging")
        dsum["is_contact"] = pc_sum.isin(contact_events)
        loc_valid = dsum["PlateLocSide"].notna() & dsum["PlateLocHeight"].notna()
        dsum["is_in_zone"] = (
            loc_valid
            & dsum["PlateLocSide"].between(ZONE_LEFT, ZONE_RIGHT, inclusive="both")
            & dsum["PlateLocHeight"].between(ZONE_BOTTOM, ZONE_TOP, inclusive="both")
        )
        dsum["is_outside"] = loc_valid & ~dsum["is_in_zone"]
        dsum["is_zone_swing"] = dsum["is_in_zone"] & dsum["is_swing"]
        dsum["is_zone_contact"] = dsum["is_in_zone"] & dsum["is_contact"]
        dsum["is_chase"] = dsum["is_outside"] & dsum["is_swing"]
        dsum["is_chase_contact"] = dsum["is_outside"] & dsum["is_contact"]

        summary = (
            dsum.groupby(PITCH_TYPE_COL, dropna=True)
            .agg(
                pitch_count=("PitchNo", "count"),
                max_velo=("RelSpeed", "max"),
                avg_velo=("RelSpeed", "mean"),
                spin_rate=("SpinRate", "mean"),
                strike_pct=("is_strike", "mean"),
                called_strike_n=("is_called_strike", "sum"),
                swings=("is_swing", "sum"),
                whiffs=("is_whiff", "sum"),
                contacts=("is_contact", "sum"),
                in_zone_n=("is_in_zone", "sum"),
                zone_swings=("is_zone_swing", "sum"),
                zone_contacts=("is_zone_contact", "sum"),
                outside_n=("is_outside", "sum"),
                chases=("is_chase", "sum"),
                chase_contacts=("is_chase_contact", "sum"),
            )
            .reset_index()
        )
        total = summary["pitch_count"].sum()
        summary["usage_pct"] = summary["pitch_count"] / total if total else 0
        summary["called_strike_pct"] = np.where(summary["pitch_count"] > 0, summary["called_strike_n"] / summary["pitch_count"], np.nan)
        summary["swing_pct"] = np.where(summary["pitch_count"] > 0, summary["swings"] / summary["pitch_count"], np.nan)
        summary["swstr_pct"] = np.where(summary["pitch_count"] > 0, summary["whiffs"] / summary["pitch_count"], np.nan)
        summary["whiff_pct"] = np.where(summary["swings"] > 0, summary["whiffs"] / summary["swings"], np.nan)
        summary["zone_swing_pct"] = np.where(summary["in_zone_n"] > 0, summary["zone_swings"] / summary["in_zone_n"], np.nan)
        summary["zone_contact_pct"] = np.where(summary["zone_swings"] > 0, summary["zone_contacts"] / summary["zone_swings"], np.nan)
        summary["chase_pct"] = np.where(summary["outside_n"] > 0, summary["chases"] / summary["outside_n"], np.nan)
        summary["chase_contact_pct"] = np.where(summary["chases"] > 0, summary["chase_contacts"] / summary["chases"], np.nan)
        summary = summary.drop(
            columns=[
                "called_strike_n", "swings", "whiffs", "contacts",
                "in_zone_n", "zone_swings", "zone_contacts",
                "outside_n", "chases", "chase_contacts",
            ]
        ).sort_values("usage_pct", ascending=False).reset_index(drop=True)

        out = summary.copy()

        out["max_velo"] = pd.to_numeric(out.get("max_velo"), errors="coerce").round(1)
        out["avg_velo"] = pd.to_numeric(out.get("avg_velo"), errors="coerce").round(1)
        out["spin_rate"] = pd.to_numeric(out.get("spin_rate"), errors="coerce").round(0)

        pct_cols = [
            "usage_pct", "strike_pct", "called_strike_pct",
            "swing_pct", "swstr_pct", "whiff_pct",
            "zone_swing_pct", "zone_contact_pct",
            "chase_pct", "chase_contact_pct"
        ]
        for col in pct_cols:
            out[col] = (pd.to_numeric(out.get(col), errors="coerce") * 100).round(1).astype(str) + "%"

        out = out.rename(columns={
            PITCH_TYPE_COL: "Pitch type",
            "pitch_count": "Count",
            "usage_pct": "Usage %",
            "max_velo": "Max Velo",
            "avg_velo": "Avg Velo",
            "spin_rate": "Spin Rate",
            "strike_pct": "Strike %",
            "called_strike_pct": "Called Strike %",
            "swing_pct": "Swing %",
            "swstr_pct": "SwStrike %",
            "whiff_pct": "Whiff %",
            "zone_swing_pct": "Zone Swing %",
            "zone_contact_pct": "Zone Contact %",
            "chase_pct": "Chase %",
            "chase_contact_pct": "Chase Contact %",
        })

        if session_type == "bullpen":
            return out[bullpen_cols]
        return out[live_scrimmage_cols]

    # ---- 7. PDF download ----
    @render.download(
        filename=lambda: f"pitch_report_{input.player() or 'unknown'}_{date.today().isoformat()}.pdf"
    )
    def download_pdf():
        selected_id = input.player()
        data = pitcher_data()
        entity_label = "Pitcher"
        entity_name_col = "Pitcher"

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        with PdfPages(tmp_path) as pdf:
            if data is None or data.empty or not selected_id:
                fig, ax = plt.subplots(figsize=(14, 9.5))
                ax.text(
                    0.5, 0.5,
                    f"No data available for the selected {entity_label.lower()}.",
                    ha="center", va="center", wrap=True,
                    fontsize=14,
                )
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
            else:
                fig = plt.figure(figsize=(16, 10))
                fig.patch.set_facecolor("#ffffff")
                gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.35])

                player_name = format_display_name(data[entity_name_col].iloc[0]) if entity_name_col in data.columns else "Unknown"
                b_side = input.batter_handedness()
                vs_suffix = ""
                if b_side == "Right":
                    vs_suffix = " vs Right-Handed Batters"
                elif b_side == "Left":
                    vs_suffix = " vs Left-Handed Batters"
                fig.suptitle(f"{entity_label} Report: {player_name}{vs_suffix}", fontsize=18, fontweight="bold", y=0.98)

                ax_pie = fig.add_subplot(gs[0, 0])
                ax_loc = fig.add_subplot(gs[0, 1])
                ax_mov = fig.add_subplot(gs[0, 2])
                ax_table = fig.add_subplot(gs[1, :])

                # Compute usage directly from selected entity's filtered data
                usage = (
                    data[data[PITCH_TYPE_COL].notna()][PITCH_TYPE_COL]
                    .value_counts()
                    .rename_axis(PITCH_TYPE_COL)
                    .reset_index(name="pitch_count")
                )
                if not usage.empty:
                    usage["usage_pct"] = usage["pitch_count"] / usage["pitch_count"].sum()
                colors = build_pitch_color_map(usage[PITCH_TYPE_COL].dropna().unique()) if not usage.empty else {}

                # 1) Usage Pie on ax_pie
                ax_pie.set_facecolor("#f7f7f7")
                if usage.empty:
                    ax_pie.text(0.5, 0.5, "No pitch usage data", ha="center", va="center", transform=ax_pie.transAxes)
                    ax_pie.axis("off")
                else:
                    labels = usage[PITCH_TYPE_COL].tolist()
                    pcts = usage["usage_pct"].values
                    c = [colors.get(pt, (0.5, 0.5, 0.5)) for pt in labels]
                    ax_pie.pie(
                        pcts,
                        labels=None,
                        colors=c,
                        startangle=90,
                        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
                        pctdistance=0.65,
                        textprops={"fontsize": 9, "fontweight": "bold"},
                    )
                    ax_pie.legend(labels, loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=8, frameon=False)
                    ax_pie.set_title("Pitch Usage", fontsize=14, fontweight="bold")

                # 2) Location Plot on ax_loc
                ax_loc.set_facecolor("#f7f7f7")
                loc = data[
                    data["PlateLocSide"].notna()
                    & data["PlateLocHeight"].notna()
                    & data[PITCH_TYPE_COL].notna()
                ]
                if loc.empty:
                    ax_loc.text(0.5, 0.5, "No pitch location data", ha="center", va="center", transform=ax_loc.transAxes)
                    ax_loc.axis("off")
                else:
                    ax_loc.add_patch(Rectangle(
                        (ZONE_LEFT - 0.3, ZONE_BOTTOM - 0.3),
                        (ZONE_RIGHT - ZONE_LEFT) + 0.6,
                        (ZONE_TOP - ZONE_BOTTOM) + 0.6,
                        facecolor="#d9d9d9",
                        edgecolor="none",
                        alpha=0.25
                    ))
                    ax_loc.add_patch(Rectangle(
                        (ZONE_LEFT, ZONE_BOTTOM),
                        ZONE_RIGHT - ZONE_LEFT,
                        ZONE_TOP - ZONE_BOTTOM,
                        fill=False,
                        linewidth=2
                    ))
                    ax_loc.plot([ZONE_LEFT, ZONE_RIGHT], [(ZONE_BOTTOM + ZONE_TOP) / 2] * 2, linestyle="--", linewidth=1, color="#1f77b4")
                    ax_loc.plot([0, 0], [ZONE_BOTTOM, ZONE_TOP], linestyle="--", linewidth=1, color="#ff7f0e")
                    for pt, g in loc.groupby(PITCH_TYPE_COL):
                        ax_loc.scatter(
                            g["PlateLocSide"],
                            g["PlateLocHeight"],
                            s=25,
                            alpha=0.8,
                            color=colors.get(pt, (0.5, 0.5, 0.5)),
                            label=pt,
                        )
                    ax_loc.add_patch(home_plate_polygon(y_front=0.10))
                    ax_loc.set_xlim(-3, 3)
                    ax_loc.set_ylim(-0.5, 5)
                    ax_loc.set_aspect("equal", adjustable="box")
                    ax_loc.set_xlabel("PlateLocSide", fontsize=9)
                    ax_loc.set_ylabel("PlateLocHeight", fontsize=9)
                    ax_loc.set_title("Pitch Locations", fontsize=14, fontweight="bold")
                    ax_loc.grid(True, alpha=0.2)
                    ax_loc.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)

                # 3) Movement Plot on ax_mov
                ax_mov.set_facecolor("#f7f7f7")
                mov = data[data[X_MOV].notna() & data[Y_MOV].notna() & data[PITCH_TYPE_COL].notna()]
                if mov.empty:
                    ax_mov.text(0.5, 0.5, "No pitch movement data", ha="center", va="center", transform=ax_mov.transAxes)
                    ax_mov.axis("off")
                else:
                    for pt, g in mov.groupby(PITCH_TYPE_COL):
                        ax_mov.scatter(
                            g[X_MOV], g[Y_MOV],
                            s=20, alpha=0.75,
                            color=colors.get(pt, (0.5, 0.5, 0.5)),
                            label=pt,
                        )
                    ax_mov.axhline(0, linewidth=1, color="#777777")
                    ax_mov.axvline(0, linewidth=1, color="#777777")
                    ax_mov.set_xlim(*MOV_XLIM)
                    ax_mov.set_ylim(*MOV_YLIM)
                    ax_mov.set_xlabel("Horizontal break (in)", fontsize=9)
                    ax_mov.set_ylabel("Induced vertical break (in)", fontsize=9)
                    ax_mov.set_title("Pitch Movements", fontsize=14, fontweight="bold")
                    ax_mov.grid(True, alpha=0.25)
                    mov_lines = []
                    for pt, g in mov.groupby(PITCH_TYPE_COL):
                        avg_hb = g[X_MOV].mean()
                        avg_ivb = g[Y_MOV].mean()
                        mov_lines.append((pt, avg_hb, avg_ivb))
                    mov_box_text = "\n".join(
                        f"{pt}: Avg HB {avg_hb:+.1f}, Avg IVB {avg_ivb:+.1f}"
                        for pt, avg_hb, avg_ivb in mov_lines
                    )
                    ax_mov.text(
                        0.98, 0.98, mov_box_text,
                        transform=ax_mov.transAxes,
                        fontsize=6.5,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.9),
                        family="monospace",
                    )
                    ax_mov.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)

                # 4) Summary Table on ax_table
                ax_table.axis("off")
                # Compute summary directly from selected entity's filtered data
                dsum = data[data[PITCH_TYPE_COL].notna()].copy()
                if dsum is None or dsum.empty:
                    summary = pd.DataFrame()
                else:
                    dsum["RelSpeed"] = pd.to_numeric(dsum["RelSpeed"], errors="coerce")
                    dsum["PlateLocSide"] = pd.to_numeric(dsum["PlateLocSide"], errors="coerce")
                    dsum["PlateLocHeight"] = pd.to_numeric(dsum["PlateLocHeight"], errors="coerce")
                    dsum["SpinRate"] = pd.to_numeric(dsum["SpinRate"], errors="coerce") if "SpinRate" in dsum.columns else np.nan
                    pc_sum = dsum["PitchCall"].astype(str).str.strip()

                    strike_events = {"StrikeCalled", "StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
                    swing_events = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
                    contact_events = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}

                    dsum["is_strike"] = pc_sum.isin(strike_events)
                    dsum["is_called_strike"] = pc_sum.eq("StrikeCalled")
                    dsum["is_swing"] = pc_sum.isin(swing_events)
                    dsum["is_whiff"] = pc_sum.eq("StrikeSwinging")
                    dsum["is_contact"] = pc_sum.isin(contact_events)
                    loc_valid = dsum["PlateLocSide"].notna() & dsum["PlateLocHeight"].notna()
                    dsum["is_in_zone"] = (
                        loc_valid
                        & dsum["PlateLocSide"].between(ZONE_LEFT, ZONE_RIGHT, inclusive="both")
                        & dsum["PlateLocHeight"].between(ZONE_BOTTOM, ZONE_TOP, inclusive="both")
                    )
                    dsum["is_outside"] = loc_valid & ~dsum["is_in_zone"]
                    dsum["is_zone_swing"] = dsum["is_in_zone"] & dsum["is_swing"]
                    dsum["is_zone_contact"] = dsum["is_in_zone"] & dsum["is_contact"]
                    dsum["is_chase"] = dsum["is_outside"] & dsum["is_swing"]
                    dsum["is_chase_contact"] = dsum["is_outside"] & dsum["is_contact"]

                    summary = (
                        dsum.groupby(PITCH_TYPE_COL, dropna=True)
                        .agg(
                            pitch_count=("PitchNo", "count"),
                            max_velo=("RelSpeed", "max"),
                            avg_velo=("RelSpeed", "mean"),
                            spin_rate=("SpinRate", "mean"),
                            strike_pct=("is_strike", "mean"),
                            called_strike_n=("is_called_strike", "sum"),
                            swings=("is_swing", "sum"),
                            whiffs=("is_whiff", "sum"),
                            contacts=("is_contact", "sum"),
                            in_zone_n=("is_in_zone", "sum"),
                            zone_swings=("is_zone_swing", "sum"),
                            zone_contacts=("is_zone_contact", "sum"),
                            outside_n=("is_outside", "sum"),
                            chases=("is_chase", "sum"),
                            chase_contacts=("is_chase_contact", "sum"),
                        )
                        .reset_index()
                    )
                    total = summary["pitch_count"].sum()
                    summary["usage_pct"] = summary["pitch_count"] / total if total else 0
                    summary["called_strike_pct"] = np.where(summary["pitch_count"] > 0, summary["called_strike_n"] / summary["pitch_count"], np.nan)
                    summary["swing_pct"] = np.where(summary["pitch_count"] > 0, summary["swings"] / summary["pitch_count"], np.nan)
                    summary["swstr_pct"] = np.where(summary["pitch_count"] > 0, summary["whiffs"] / summary["pitch_count"], np.nan)
                    summary["whiff_pct"] = np.where(summary["swings"] > 0, summary["whiffs"] / summary["swings"], np.nan)
                    summary["zone_swing_pct"] = np.where(summary["in_zone_n"] > 0, summary["zone_swings"] / summary["in_zone_n"], np.nan)
                    summary["zone_contact_pct"] = np.where(summary["zone_swings"] > 0, summary["zone_contacts"] / summary["zone_swings"], np.nan)
                    summary["chase_pct"] = np.where(summary["outside_n"] > 0, summary["chases"] / summary["outside_n"], np.nan)
                    summary["chase_contact_pct"] = np.where(summary["chases"] > 0, summary["chase_contacts"] / summary["chases"], np.nan)
                    summary = summary.drop(
                        columns=[
                            "called_strike_n", "swings", "whiffs", "contacts",
                            "in_zone_n", "zone_swings", "zone_contacts",
                            "outside_n", "chases", "chase_contacts",
                        ]
                    ).sort_values("usage_pct", ascending=False).reset_index(drop=True)
                if summary is None or summary.empty:
                    ax_table.text(0.5, 0.5, "No summary metrics available.", ha="center", va="center", fontsize=12)
                else:
                    out = summary.copy()
                    out["max_velo"] = pd.to_numeric(out.get("max_velo"), errors="coerce").round(1)
                    out["avg_velo"] = pd.to_numeric(out.get("avg_velo"), errors="coerce").round(1)
                    out["spin_rate"] = pd.to_numeric(out.get("spin_rate"), errors="coerce").round(0)

                    pct_cols = [
                        "usage_pct", "strike_pct", "called_strike_pct",
                        "swing_pct", "swstr_pct", "whiff_pct",
                        "zone_swing_pct", "zone_contact_pct",
                        "chase_pct", "chase_contact_pct"
                    ]
                    for col in pct_cols:
                        out[col] = (pd.to_numeric(out.get(col), errors="coerce") * 100).round(1).astype(str) + "%"

                    display_df = out.rename(columns={
                        PITCH_TYPE_COL: "Pitch",
                        "pitch_count": "Count",
                        "usage_pct": "Usage%",
                        "max_velo": "MaxV",
                        "avg_velo": "AvgV",
                        "spin_rate": "Spin",
                        "strike_pct": "K%",
                        "called_strike_pct": "CalledK%",
                        "swing_pct": "Swing%",
                        "swstr_pct": "SwStr%",
                        "whiff_pct": "Whiff%",
                        "zone_swing_pct": "ZoneSw%",
                        "zone_contact_pct": "ZoneCon%",
                        "chase_pct": "Chase%",
                        "chase_contact_pct": "ChaseCon%",
                    })
                    session_type = "live"
                    if "SessionType" in data.columns:
                        st = str(data["SessionType"].iloc[0]).strip().lower()
                        if st == "bullpen":
                            session_type = "bullpen"

                    bullpen_cols = ["Pitch", "Count", "Usage%", "MaxV", "AvgV", "Spin", "K%"]
                    live_cols = [
                        "Pitch", "Count", "Usage%", "MaxV", "AvgV", "Spin", "K%",
                        "CalledK%", "Swing%", "SwStr%", "Whiff%", "ZoneSw%",
                        "ZoneCon%", "Chase%", "ChaseCon%"
                    ]
                    table_cols = bullpen_cols if session_type == "bullpen" else live_cols
                    display_df = display_df[table_cols]

                    table = ax_table.table(
                        cellText=display_df.values,
                        colLabels=display_df.columns,
                        loc="center",
                        cellLoc="center",
                    )
                    table.auto_set_font_size(False)
                    n_rows = len(display_df)
                    if n_rows <= 6:
                        table.set_fontsize(8)
                        table.scale(1.0, 1.35)
                    elif n_rows <= 10:
                        table.set_fontsize(7)
                        table.scale(1.0, 1.15)
                    else:
                        table.set_fontsize(6)
                        table.scale(1.0, 0.98)

                    # Style table cells: header and zebra striping
                    for (row, col), cell in table.get_celld().items():
                        if row == 0:  # Header row
                            cell.set_facecolor("#111111")
                            cell.set_text_props(color="white", fontweight="bold")
                        else:  # Body rows - zebra striping
                            if row % 2 == 1:  # Odd rows
                                cell.set_facecolor("#f3f3f3")
                            else:  # Even rows
                                cell.set_facecolor("#ffffff")
                        cell.set_edgecolor("#cfcfcf")

                    ax_table.set_title("Pitch Summary Metrics", fontsize=14, fontweight="bold", pad=10)

                # Apply tight layout with spacing tuned for legends and larger table
                fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.2, w_pad=3.8)

                pdf.savefig(fig)
                plt.close(fig)

        # Read the PDF as binary bytes and yield them, then clean up
        with open(tmp_path, "rb") as f:
            yield f.read()

        os.remove(tmp_path)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = App(app_ui, server, static_assets=STATIC_DIR)