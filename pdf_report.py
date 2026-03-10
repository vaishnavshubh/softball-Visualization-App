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
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
from shiny import App, render, ui, reactive


# ---------------------------------------------------------------------------
# Config / constants (mirrors app.py where needed)
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

FIG_SIZE = (3.6, 2.8)

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
    "Fastball":  "#1F3A5F",
    "Changeup":  "#E0B84C",
    "Curveball": "#4E6E81",
    "Riseball":  "#8FA6B3",
    "Dropball":  "#5B6770",
    "Screwball": "#A7B1B7",
    "Offspeed":  "#C7CED3",
    "Riser": "#8FA6B3",
    "Rise":  "#8FA6B3",
    "Drop":  "#5B6770",
}
PITCH_TYPE_FALLBACK_COLORS = ["#6C7A89", "#9AA5AD", "#B8C2C8", "#D3D9DE"]


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
    """
    Strike %:
      (StrikeCalled + StrikeSwinging + Fouls + InPlay) / total pitches
    Whiff %:
      StrikeSwinging / swings
      swings = StrikeSwinging + Fouls + InPlay
    """
    if df is None or df.empty or pitcher_id is None:
        return pd.DataFrame()

    required = {"PitcherId", PITCH_TYPE_COL, "PitchNo", "RelSpeed", "PitchCall", "PlateLocSide", "PlateLocHeight"}
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    d = df[
        (df["PitcherId"].astype(str) == str(pitcher_id)) &
        df[PITCH_TYPE_COL].notna()
    ].copy()

    if d.empty:
        return pd.DataFrame()

    d["RelSpeed"] = pd.to_numeric(d["RelSpeed"], errors="coerce")
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

    d["PlateLocSide"]   = pd.to_numeric(d["PlateLocSide"], errors="coerce")
    d["PlateLocHeight"] = pd.to_numeric(d["PlateLocHeight"], errors="coerce")

    d["is_strike"] = pc.isin(STRIKE_EVENTS)
    d["is_swing"]  = pc.isin(SWING_EVENTS)
    d["is_whiff"]  = pc.eq("StrikeSwinging")

    d["is_outside"] = (
        (d["PlateLocSide"]   < ZONE_LEFT)  |
        (d["PlateLocSide"]   > ZONE_RIGHT) |
        (d["PlateLocHeight"] < ZONE_BOTTOM)|
        (d["PlateLocHeight"] > ZONE_TOP)
    )
    d["is_chase"] = d["is_outside"] & d["is_swing"]

    summary = (
        d.groupby(PITCH_TYPE_COL, dropna=True)
        .agg(
            pitch_count=("PitchNo", "count"),
            max_velo=("RelSpeed", "max"),
            avg_velo=("RelSpeed", "mean"),
            spin_rate=("SpinRate", "mean"),
            strike_pct=("is_strike", "mean"),
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum"),
            outside=("is_outside", "sum"),
            chases=("is_chase", "sum"),
        )
        .reset_index()
    )

    summary["whiff_pct"] = np.where(
        summary["swings"] > 0,
        summary["whiffs"] / summary["swings"],
        np.nan
    )

    summary["chase_pct"] = np.where(
        summary["outside"] > 0,
        summary["chases"] / summary["outside"],
        np.nan
    )

    total = summary["pitch_count"].sum()
    summary["usage_pct"] = summary["pitch_count"] / total if total else 0

    return (
        summary
        .drop(columns=["swings", "whiffs", "outside", "chases"])
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


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.h2("Softball Pitch Dashboard (Uploaded CSV)"),

    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file(
                "file",
                "Upload CSV (TrackMan schema)",
                multiple=False,
                accept=[".csv"],
            ),

            ui.output_ui("date_range_ui"),

            ui.input_select(
                "team",
                "Team Name",
                choices={},
            ),
            ui.input_select(
                "session_type",
                "Session Type",
                choices={
                    "all": "All",
                    "bullpen": "Bullpen",
                    "bp": "Batting Practice",
                    "scrimmage": "Scrimmage",
                },
                selected="all",
            ),
            ui.input_radio_buttons(
                "player_type",
                "Player Type",
                choices={"pitcher": "Pitcher", "batter": "Batter"},
                selected="pitcher",
            ),
            ui.input_select("player", "Player Name", choices={"": "—"}),
            width=3,
        ),
        ui.panel_main(
            ui.output_ui("main_tabs")
        ),
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
                df[c] = df[c].astype(str).str.strip()

        df["SessionType"] = infer_session_type_for_purdue(df)
        return df

    # ---- 2. Date bounds & date filter ----
    @reactive.calc
    def date_bounds():
        df = uploaded_df()
        if df is None or "Date" not in df.columns or df["Date"].empty:
            return None, None
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if dates.empty:
            return None, None
        return dates.min().date(), dates.max().date()

    @output
    @render.ui
    def date_range_ui():
        dmin, dmax = date_bounds()
        if dmin is None or dmax is None:
            return ui.div("Upload a CSV to enable date filtering.")
        return ui.div(
            ui.input_date("date_start", "Start", value=dmin, min=dmin, max=dmax),
            ui.input_date("date_end", "End", value=dmax, min=dmin, max=dmax),
        )

    @reactive.calc
    def date_range_invalid():
        start = input.date_start()
        end = input.date_end()
        return start is not None and end is not None and start > end

    @reactive.calc
    def current_df():
        df = uploaded_df()
        if df is None:
            return None

        dmin, dmax = date_bounds()
        start = input.date_start()
        end = input.date_end()

        if start is None or end is None or dmin is None or dmax is None:
            return df

        if start > end:
            return None

        df = df.copy()
        df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df = df[df["_date"].notna() & (df["_date"] >= start) & (df["_date"] <= end)]
        df = df.drop(columns=["_date"], errors="ignore")
        if df.empty:
            return None
        return df

    # ---- 3. Team & player dropdowns ----
    @reactive.effect
    def _update_team_choices():
        df = current_df()
        if df is None or df.empty:
            ui.update_select("team", choices={}, session=session)
            return

        teams = set()
        for col in ["PitcherTeam", "BatterTeam"]:
            if col in df.columns:
                vals = df[col].dropna().astype(str).str.strip().tolist()
                teams.update([v for v in vals if v])

        teams = sorted(list(teams))
        choices = {t: ("Purdue University" if t == PURDUE_CODE else t) for t in teams}
        default_team = PURDUE_CODE if PURDUE_CODE in teams else (teams[0] if teams else None)
        ui.update_select("team", choices=choices, selected=default_team, session=session)

    @reactive.effect
    def _force_session_all_for_non_purdue():
        team = input.team()
        if team and not is_purdue_team(team):
            ui.update_select("session_type", selected="all", session=session)

    @reactive.effect
    def _update_player_choices():
        df = current_df()
        team = input.team()
        ptype = input.player_type()

        if df is None or df.empty or not team:
            ui.update_select("player", choices={"": "—"}, label="Player Name", session=session)
            return

        if ptype == "pitcher":
            if "PitcherTeam" in df.columns:
                df = df[df["PitcherTeam"].astype(str).str.strip() == str(team)]
            if "PitcherId" not in df.columns or "Pitcher" not in df.columns:
                ui.update_select("player", choices={"": "—"}, label="Player Name", session=session)
                return
            lookup = (
                df[["PitcherId", "Pitcher"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["Pitcher", "PitcherId"])
            )
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
        else:
            # Batter mode placeholder; you can mirror pitcher logic later.
            ui.update_select("player", choices={"": "—"}, label="Player Name", session=session)

    # ---- 4. Core reactive data for visuals ----
    @reactive.calc
    def pitcher_data():
        df = current_df()
        team = input.team()
        pid = input.player() if input.player_type() == "pitcher" else None
        if df is None or not team or not pid:
            return None

        if "PitcherTeam" in df.columns:
            df = df[df["PitcherTeam"].astype(str).str.strip() == str(team)]

        df = apply_session_filter_for_team(df, team, input.session_type())
        return df[df["PitcherId"].astype(str) == str(pid)]

    @reactive.calc
    def pitch_colors():
        data = pitcher_data()
        if data is None or PITCH_TYPE_COL not in data.columns:
            return {}
        return build_pitch_color_map(data[PITCH_TYPE_COL].dropna().unique())

    @reactive.calc
    def usage_df():
        data = pitcher_data()
        pid = input.player() if input.player_type() == "pitcher" else None
        if data is None or not pid or input.player_type() != "pitcher":
            return pd.DataFrame(columns=[PITCH_TYPE_COL, "pitch_count", "usage_pct"])
        return compute_usage(data, pid)

    @reactive.calc
    def player_summary_text():
        if input.player_type() != "pitcher":
            return "Select Pitcher to view profile"

        data = pitcher_data()
        if data is None or data.empty:
            return "Select Pitcher to view profile"

        name_raw = data["Pitcher"].iloc[0] if "Pitcher" in data.columns else ""
        throws_raw = data["PitcherThrows"].iloc[0] if "PitcherThrows" in data.columns else ""
        name = format_display_name(name_raw) or "Pitcher"
        hand = throws_to_short(throws_raw)

        n_pitches = len(data)

        if hand:
            return f"{name} | {hand} {n_pitches} pitches"
        return f"{name} {n_pitches} pitches"

    # ---- 5. UI tabs ----
    @output
    @render.ui
    def main_tabs():
        if date_range_invalid():
            return ui.div("Start date must be on or before end date.")

        df = current_df()
        if df is None or df.empty:
            return ui.div("Upload a CSV (with the standard schema) and select a team/player.")

        return ui.navset_tab(
            ui.nav_panel(
                "Home",
                ui.div(
                    ui.h3("Pitch Profile"),
                    ui.div(ui.output_text("player_summary")),
                    ui.output_ui("movement_legend"),
                    ui.layout_columns(
                        ui.output_plot("pie"),
                        ui.output_plot("location"),
                    ),
                    ui.layout_columns(
                        ui.panel_well(ui.output_table("usage_table")),
                        ui.output_plot("movement"),
                    ),
                ),
            ),
            ui.nav_panel(
                "Development",
                ui.div(
                    ui.h3("Strike & Whiff Trends"),
                    ui.output_plot("dev_strike_whiff_trend"),
                ),
            ),
        )

    @output
    @render.text
    def player_summary():
        return player_summary_text()

    # ---- 6. Plots & table (same logic as app.py) ----
    @output
    @render.plot
    def pie():
        usage = usage_df()
        colors = pitch_colors()

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
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
            textprops={"fontsize": 10, "fontweight": "bold"},
        )
        for t in ax.texts:
            t.set_fontsize(8)

        ax.set_title("Pitch Usage", fontsize=16, fontweight="bold")
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
            alpha=0.25
        ))
        ax.add_patch(Rectangle(
            (ZONE_LEFT, ZONE_BOTTOM),
            ZONE_RIGHT - ZONE_LEFT,
            ZONE_TOP - ZONE_BOTTOM,
            fill=False,
            linewidth=2
        ))
        ax.plot([ZONE_LEFT, ZONE_RIGHT], [(ZONE_BOTTOM + ZONE_TOP) / 2] * 2, linestyle="--", linewidth=1)
        ax.plot([0, 0], [ZONE_BOTTOM, ZONE_TOP], linestyle="--", linewidth=1)

        for pt, g in loc.groupby(PITCH_TYPE_COL):
            ax.scatter(
                g["PlateLocSide"],
                g["PlateLocHeight"],
                s=35,
                alpha=0.8,
                color=colors.get(pt, (0.5, 0.5, 0.5)),
            )

        ax.add_patch(home_plate_polygon(y_front=0.10))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PlateLocSide")
        ax.set_ylabel("PlateLocHeight")
        ax.set_title("Pitch Locations", fontsize=16, fontweight="bold")
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
                s=25, alpha=0.75,
                color=colors.get(pt, (0.5, 0.5, 0.5))
            )

        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)
        ax.set_xlim(*MOV_XLIM)
        ax.set_ylim(*MOV_YLIM)
        ax.set_xlabel("Horizontal break (in)")
        ax.set_ylabel("Induced vertical break (in)")
        ax.set_title("Pitch Movements", fontsize=16, fontweight="bold")
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
            fontsize=7,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.85),
            family="monospace",
        )

        return fig

    import math

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

        return ui.div(*items)

    @output
    @render.plot
    def dev_strike_whiff_trend():
        if input.player_type() != "pitcher":
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "Pitcher view only.", ha="center", va="center")
            ax.set_axis_off()
            return fig

        data = pitcher_data()
        if data is None or data.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No data for selected filters.", ha="center", va="center")
            ax.set_axis_off()
            return fig

        if "Date" not in data.columns or PITCH_TYPE_COL not in data.columns or "PitchCall" not in data.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "Missing Date / Pitch Type / PitchCall columns.", ha="center", va="center")
            ax.set_axis_off()
            return fig

        df = data.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"] = df["Date"].dt.normalize()
        df = df.dropna(subset=["Date", PITCH_TYPE_COL])

        pc = df["PitchCall"].astype(str).str.strip()

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

        df["is_strike"] = pc.isin(STRIKE_EVENTS)
        df["is_swing"] = pc.isin(SWING_EVENTS)
        df["is_whiff"] = pc.eq("StrikeSwinging")

        g = (
            df.groupby([pd.Grouper(key="Date", freq="D"), PITCH_TYPE_COL])
              .agg(
                  pitch_n=("is_strike", "size"),
                  strike_pct=("is_strike", "mean"),
                  swings=("is_swing", "sum"),
                  whiffs=("is_whiff", "sum"),
              )
              .reset_index()
              .sort_values([PITCH_TYPE_COL, "Date"])
        )

        g["whiff_pct"] = np.where(g["swings"] > 0, g["whiffs"] / g["swings"], np.nan)
        g = g[g["pitch_n"] >= 8].copy()
        g = g[g[PITCH_TYPE_COL].astype(str).str.lower() != "other"].copy()

        if g.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "Not enough pitch samples to plot trends.", ha="center", va="center")
            ax.set_axis_off()
            return fig

        pitch_types = list(g[PITCH_TYPE_COL].dropna().unique())
        colors = pitch_colors()

        n = len(pitch_types)
        ncols = 2 if n > 1 else 1
        nrows = math.ceil(n / ncols)

        fig_w = 14
        fig_h = 5 * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

        if nrows == 1 and ncols == 1:
            axes_list = [axes]
        elif nrows == 1:
            axes_list = list(axes)
        else:
            axes_list = [ax for row in axes for ax in (row if isinstance(row, (list, np.ndarray)) else [row])]

        fig.patch.set_facecolor("#ffffff")
        fig.subplots_adjust(top=0.82, hspace=0.5, wspace=0.3)

        for i, pt in enumerate(pitch_types):
            ax = axes_list[i]
            sub = g[g[PITCH_TYPE_COL] == pt]

            c = colors.get(pt, "#1F3A5F")
            n_total = int(sub["pitch_n"].sum())

            ax.plot(
                sub["Date"], sub["strike_pct"] * 100,
                linewidth=2.5, marker="o", markersize=7,
                color="#DDB945", label="Strike %", zorder=3
            )
            ax.plot(
                sub["Date"], sub["whiff_pct"] * 100,
                linewidth=2.5, linestyle="--", marker="s", markersize=6,
                color="#AAAAAA", label="Whiff %", zorder=3
            )

            ax.set_xlim(sub["Date"].min(), sub["Date"].max())
            ax.set_title(f"{pt}  (n={n_total})", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=25)

            if ncols == 1 or (i % ncols == 0):
                ax.set_ylabel("Percent", fontsize=10)

        for j in range(n, len(axes_list)):
            axes_list[j].set_facecolor("#f5f5f5")
            axes_list[j].set_axis_off()

        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        formatter = mdates.DateFormatter("%m-%d")
        for ax in axes_list[:n]:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        legend_handles = [
            Line2D([0], [0], color="#DDB945", linewidth=2.5, marker="o", markersize=7, label="Strike %"),
            Line2D([0], [0], color="#AAAAAA", linewidth=2.5, linestyle="--", marker="s", markersize=6, label="Whiff %"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2,
                   frameon=True, fancybox=True, edgecolor="#cccccc",
                   bbox_to_anchor=(0.5, 1.015), fontsize=11)

        return fig

    @output
    @render.table
    def usage_table():
        data = pitcher_data()
        pid = input.player() if input.player_type() == "pitcher" else None

        cols = ["Pitch type", "Count", "Usage %", "Max Velo", "Avg Velo", "Spin Rate", "Strike %", "Whiff %", "Chase %"]

        if data is None or data.empty or not pid:
            return pd.DataFrame(columns=cols)

        summary = compute_pitch_metrics(data, pid)
        if summary is None or summary.empty:
            return pd.DataFrame(columns=cols)

        out = summary.copy()

        out["max_velo"] = pd.to_numeric(out.get("max_velo"), errors="coerce").round(1)
        out["avg_velo"] = pd.to_numeric(out.get("avg_velo"), errors="coerce").round(1)
        out["spin_rate"] = pd.to_numeric(out.get("spin_rate"), errors="coerce").round(0)
        out["usage_pct"] = (out["usage_pct"] * 100).round(1).astype(str) + "%"

        out["strike_pct"] = (out["strike_pct"] * 100).round(1).astype(str) + "%"
        out["whiff_pct"] = (out["whiff_pct"] * 100).round(1).astype(str) + "%"
        out["chase_pct"]  = (out["chase_pct"]  * 100).round(1).astype(str) + "%"

        out = out.rename(columns={
            PITCH_TYPE_COL: "Pitch type",
            "pitch_count": "Count",
            "usage_pct": "Usage %",
            "max_velo": "Max Velo",
            "avg_velo": "Avg Velo",
            "spin_rate": "Spin Rate",
            "strike_pct": "Strike %",
            "whiff_pct": "Whiff %",
            "chase_pct": "Chase %"
        })

        return out[cols]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = App(app_ui, server)