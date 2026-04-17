"""
Softball Pitch Dashboard - Shiny for Python

Layout (matches sketch):
- Left sidebar: Filters
- Right main:
  - Tabs: Home, Comparison, Development
  - Home shows:
      Pitch Profile header
      Player summary line
      2x2 grid: Usage pie, Location, Summary table, Movement
"""

import glob
import json
import math
import os
import sys
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, Ellipse
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from shiny import App, render, ui, reactive

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
import prediction_pipeline  # noqa: E402  # league-wide ML for Prediction tab (after APP_DIR on path)

V3_PATH = os.path.join(APP_DIR, "data", "v3")
STATIC_DIR = os.path.join(APP_DIR, "static")

#Team name map
TEAM_NAME_MAP = {
    "UNI_KAN_SB": "University of Kansas",
    "PUR_BOI_SB": "Purdue University",
    "TEX_STA_SB": "Texas State University",
    "UNI_IOW_SB": "University of Iowa",
    "UNI_ARK_SB": "University of Arkansas",
    "NEB_COR_SB": "University of Nebraska",
    "BOS_COL_SB": "Boston College",
    "CRE_UNI_SB": "Creighton University",
    "IOW_STA_SB": "Iowa State University",
    "STA_WOL_SB": "North Carolina State University",
    "LOU_TEC_SB": "Louisiana Tech University",
    "DUK_BLU_SB": "Duke University",
    "UNI_UTA_SB": "University of Utah",
    "ILL_FIG_SB": "University of Illinois",
    "FLO_GAT_SB": "University of Florida",
    "CAL_BAP_SB": "California Baptist University",
    "BYU_COU_SB": "Brigham Young University",
    "GEO_BUL_SB": "University of Georgia",
    "NOR_GEO_SB": "University of North Georgia",
    "ARI_WIL_SB": "University of Arizona",
    "AUB_TIG_SB": "Auburn University",
    "MIN_GOL_SB": "University of Minnesota",
    "UNI_KEN_SB": "University of Kentucky",
    "LON_BEA_SB": "Long Beach State University",
    "LSU_TIG_SB": "Louisiana State University",
    "CLE_TIG_SB": "Clemson University",
    "OKL_SOO_SB": "University of Oklahoma",
    "OKL_STA_SB": "Oklahoma State University",
    "ORE_STA_SB": "Oregon State University",
    "VIR_CAV_SB": "University of Virginia",
    "TEX_TEC_SB": "Texas Tech University",
    "SAN_DIE1_SB": "San Diego State University",
    "SAN_DIE2_SB": "University of San Diego",
    "UNI_TEN_SB": "University of Tennessee",
    "STA_UNI_SB": "Stanford University",
    "VIR_TEC_SB": "Virginia Tech",
    "TEX_A&M_SB": "Texas A&M University",
    "MEM_TIG_SB": "University of Memphis",
    "IND_HOO_SB": "Indiana University",
    "CHA_49E_SB": "University of North Carolina at Charlotte",
    "OMA_MAV_SB": "University of Nebraska Omaha",
    "WAS_HUS_SB": "University of Washington",
    "TEX_LON_SB": "University of Texas",
    "WIC_STA_SB": "Wichita State University",
    "ABI_CHR_SB": "Abilene Christian University",
    "AKR_ZIP_SB": "University of Akron",
    "ALA_A&M_SB": "Alabama A&M University",
    "ALA_CRI_SB": "University of Alabama",
    "ALC_STA_SB": "Alcorn State University",
    "AND_UNI_SB": "Anderson University",
    "APP_STA_SB": "Appalachian State University",
    "ARI_STA_SB": "Arizona State University",
    "AUG_UNI_SB": "Augusta University",
    "AUS_PEA_SB": "Austin Peay State University",
    "BAL_STA_SB": "Ball State University",
    "BAR_UNI_SB": "Barry University",
    "BAY_BEA_SB": "Baylor University",
    "BEL_UNI_SB": "Belhaven University",
    "BEL_BEA_SB": "Belmont University",
    "BET_COO_SB": "Bethune-Cookman University",
    "BIN_UNI_SB": "Binghamton University",
    "BOI_STA_SB": "Boise State University",
    "BOS_UNI_SB": "Boston University",
    "BRA_BRA_SB": "Bradley University",
    "BRO_UNI_SB": "Brown University",
    "BRY_UNI_SB": "Bryant University",
    "BUC_UNI_SB": "Bucknell University",
    "BUT_COM_SB": "Butler Community College",
    "BUT_UNI_SB": "Butler University",
    "CAL_POL_SB": "California Polytechnic State University",
    "CAL_STA_SB": "California State University, Fullerton",
    "CAL_GOL_SB": "University of California, Berkeley",
    "CAL_STA1_SB": "California State University, Northridge",
    "CAM_FIG_SB": "Campbell University",
    "CAR_NEW_SB": "Carson-Newman University",
    "CEN_ARK_SB": "University of Central Arkansas",
    "CEN_MIC_SB": "Central Michigan University",
    "CHI_COL_SB": "Chipola College",
    "COA_CAR_SB": "Coastal Carolina University",
    "COL_CHA_SB": "College of Charleston",
    "COL_CHR_SB": "Colorado Christian University",
    "COL_STA_SB": "Colorado State University",
    "COP_COP_SB": "Copiah-Lincoln Community College",
    "CSU_BAK_SB": "California State University, Bakersfield",
    "DEP_BLU_SB": "DePaul University",
    "DET_MER_SB": "University of Detroit Mercy",
    "DRA_UNI_SB": "Drake University",
    "EAS_CAR_SB": "East Carolina University",
    "EAS_TEN_SB": "East Tennessee State University",
    "EAS_TEX_SB": "East Texas A&M University",
    "EAS_ILL_SB": "Eastern Illinois University",
    "ELO_UNI_SB": "Elon University",
    "EMB_RID_SB": "Embry-Riddle Aeronautical University",
    "EMM_UNI_SB": "Emmanuel University",
    "EMP_STA_SB": "Emporia State University",
    "EVA_PUR_SB": "University of Evansville",
    "FLA_UNI_SB": "Flagler College",
    "FLO_ATL_SB": "Florida Atlantic University",
    "FLO_GUL_SB": "Florida Gulf Coast University",
    "FLO_INT_SB": "Florida International University",
    "FLO_STA_SB": "Florida State University",
    "FOR_UNI_SB": "Fordham University",
    "FRE_STA_SB": "California State University, Fresno",
    "FUR_PAL_SB": "Furman University",
    "GAR_CIT_SB": "Garden City Community College",
    "GAR_RUN_SB": "Gardner-Webb University",
    "GEO_WAS1_SB": "George Washington University",
    "GEO_HOY_SB": "Georgetown University",
    "GEO_GWI_SB": "Georgia Gwinnett College",
    "GEO_SOU_SB": "Georgia Southern University",
    "GEO_STA_SB": "Georgia State University",
    "GEO_TEC_SB": "Georgia Institute of Technology",
    "GRA_CAN_SB": "Grand Canyon University",
    "GRA_JUN_SB": "Grand Junction",
    "GRA_JUN1_SB": "Grand Junction Central",
    "GRE_BAY_SB": "University of Wisconsin-Green Bay",
    "HAW_RAI_SB": "University of Hawaiʻi",
    "HOW_UNI_SB": "Howard University",
    "HUT_COM_SB": "Hutchinson Community College",
    "IDA_STA_SB": "Idaho State University",
    "ILL_STA_SB": "Illinois State University",
    "INC_WOR_SB": "University of the Incarnate Word",
    "IND_STA_SB": "Indiana State University",
    "IND_UNI_SB": "Indiana University Indianapolis",
    "JAC_STA_SB": "Jacksonville State University",
    "JAC_UNI_SB": "Jacksonville University",
    "JAM_MAD_SB": "James Madison University",
    "JON_COL_SB": "Jones College",
    "KEN_STA_SB": "Kennesaw State University",
    "LAM_UNI_SB": "Lamar University",
    "LEE_UNI_SB": "Lee University",
    "LEH_UNI_SB": "Lehigh University",
    "LIB_LAD_SB": "Liberty University",
    "LIN_MEM_SB": "Lincoln Memorial University",
    "LIN_LIO_SB": "Lindenwood University",
    "LIP_LIP_SB": "Lipscomb University",
    "LIU_SHA_SB": "Long Island University",
    "LON_LAN_SB": "Longwood University",
    "LOU_COL_SB": "Louisburg College",
    "LOU_RAG_SB": "University of Louisiana",
    "LOU_CAR_SB": "University of Louisville",
    "LOY_CHI_SB": "Loyola University Chicago",
    "LOY_MAR_SB": "Loyola Marymount University",
    "MAR_UNI_SB": "Marist University",
    "MAR_THU_SB": "Marshall University",
    "MAR_TER_SB": "University of Maryland",
    "MCL_COL_SB": "McLennan Community College",
    "MCN_STA1_SB": "McNeese State University",
    "MER_BEA_SB": "Mercer University",
    "MET_STA_SB": "Metropolitan State University of Denver",
    "MIA_(OH_SB": "Miami Ohio University",
    "MIC_STA_SB": "Michigan State University",
    "MIC_WOL_SB": "University of Michigan",
    "MID_TEN_SB": "Middle Tennessee State University",
    "MIS_STA_SB": "Mississippi State University",
    "MIS_STA1_SB": "Missouri State University",
    "MIS_TIG_SB": "University of Missouri",
    "MON_GRI_SB": "University of Montana",
    "MUR_STA_SB": "Murray State College",
    "NEW_MEX1_SB": "University of New Mexico",
    "NEW_MEX_SB": "New Mexico State University",
    "NIC_STA_SB": "Nicholls State University",
    "NOR_ALA_SB": "University of North Alabama",
    "NOR_CAR2_SB": "North Carolina A&T State University",
    "NOR_CAR1_SB": "North Carolina Central University",
    "NOR_CAR_SB": "University of North Carolina",
    "NOR_DAK1_SB": "University of North Dakota",
    "NOR_DAK_SB": "North Dakota State University",
    "NOR_ILL_SB": "Northern Illinois University",
    "NOR_FLO_SB": "Northwest Florida State College",
    "NOR_STA_SB": "Northwestern State University",
    "NOR_WIL_SB": "Northwestern University",
    "NOR_UNI_SB": "Northwood University",
    "OHI_STA_SB": "Ohio State University",
    "OKL_CHR_SB": "Oklahoma Christian University",
    "OLE_MIS_SB": "University of Mississippi",
    "ORE_DUC_SB": "University of Oregon",
    "PEN_STA_SB": "Pennsylvania State University",
    "PRA_VIE_SB": "Prairie View A&M University",
    "PRE_BLU_SB": "Presbyterian College",
    "PRI_UNI_SB": "Princeton University",
    "PRO_FRI_SB": "Providence College",
    "QUE_UNI_SB": "Queens University of Charlotte",
    "RAD_UNI_SB": "Radford University",
    "ROA_STA_SB": "Roane State Community College",
    "RUT_RUT_SB": "Rutgers University",
    "SAI_FRA_SB": "Saint Francis University",
    "SAI_LEO_SB": "Saint Leo University",
    "SAI_MAR_SB": "Saint Mary's College",
    "SAL_LAK_SB": "Salt Lake Community College",
    "SAM_HOU_SB": "Sam Houston State University",
    "SAM_UNI_SB": "Samford University",
    "SAN_JOS_SB": "San Jose State University",
    "SAN_CLA_SB": "Santa Clara University",
    "SEA_UNI_SB": "Seattle University",
    "SEM_STA_SB": "Seminole State College",
    "SET_HAL_SB": "Seton Hall University",
    "SHO_UNI_SB": "Shorter University",
    "SOU_ALA_SB": "University of South Alabama",
    "SOU_CAR_SB": "University of South Carolina",
    "SOU_DAK_SB": "South Dakota State University",
    "SOU_FLO_SB": "University of South Florida",
    "SOU_MIS1_SB": "Southeast Missouri State University",
    "SOU_LOU_SB": "Southeastern Louisiana University",
    "SOU_ILL_SB": "Southern Illinois University Edwardsville",
    "SOU_ILL1_SB": "Southern Illinois University",
    "SOU_IND_SB": "University of Southern Indiana",
    "SOU_MIS_SB": "University of Southern Mississippi",
    "SOU_NAZ_SB": "Southern Nazarene University",
    "SOU_UNI_SB": "Southern University",
    "SOU_UTA_SB": "Southern Utah University",
    "ST._BON_SB": "St. Bonaventure University",
    "STE_AUS_SB": "Stephen F. Austin State University",
    "STE_HAT_SB": "Stetson University",
    "SYR_ORA_SB": "Syracuse University",
    "TEM_COL_SB": "Temple College",
    "TEN_TEC_SB": "Tennessee Tech University",
    "TEX_A&M2_SB": "Texas A&M University-Corpus Christi",
    "TRO_TRO_SB": "Troy University",
    "TUL_GOL_SB": "University of Tulsa",
    "TUS_UNI_SB": "Tusculum University",
    "RIV_RIV_SB": "University of California, Riverside",
    "SAN_DIE_SB": "University of California, San Diego",
    "SAN_BAR_SB": "University of California, Santa Barbara",
    "UCF_KNI_SB": "University of Central Florida",
    "UCL_BRU_SB": "University of California, Los Angeles",
    "UCO_HUS_SB": "University of Connecticut",
    "UIC_UIC_SB": "University of Illinois Chicago",
    "UNC_WIL_SB": "University of North Carolina Wilmington",
    "UNI_ALA_SB": "University of Alabama at Birmingham",
    "UNI_DEL_SB": "University of Delaware",
    "UNI_HOU_SB": "University of Houston",
    "UNI_LOU_SB": "University of Louisiana Monroe",
    "UNI_MAS_SB": "University of Massachusetts",
    "UNI_MIS_SB": "University of Missouri-Kansas City",
    "UNI_NEV_SB": "University of Nevada",
    "UNI_NOR2_SB": "University of North Florida",
    "UNI_NOR1_SB": "University of North Texas",
    "UNI_NOR3_SB": "University of Northern Iowa",
    "DAV_AGG_SB": "UC Davis Aggies",
    "HAM_UNI_SB": "Hampton University",
    "UNI_NOT_SB": "University of Notre Dame",
    "UNI_SAN_SB": "University of San Diego",
    "UNI_SOU_SB": "University of South Dakota",
    "UNI_ST.2_SB": "University of St. Thomas",
    "UNI_TEX1_SB": "University of Texas at El Paso",
    "UNI_TEX2_SB": "University of Texas at San Antonio",
    "UNI_TOL_SB": "University of Toledo",
    "UNI_WES_SB": "University of West Georgia",
    "UNL_UNL_SB": "University of Nevada, Las Vegas",
    "USC_AIK_SB": "University of South Carolina Aiken",
    "USC_BEA_SB": "University of South Carolina Beaufort",
    "USC_UPS_SB": "University of South Carolina Upstate",
    "ARL_MAV_SB": "University of Texas at Arlington",
    "UTA_STA_SB": "Utah State University",
    "UTA_TEC_SB": "Utah Tech University",
    "UTA_VAL_SB": "Utah Valley University",
    "VAL_STA_SB": "Valdosta State University",
    "VAL_VAL_SB": "Valparaiso University",
    "VIL_WIL_SB": "Villanova University",
    "WAL_STA1_SB": "Wallace State Community College",
    "WAL_STA_SB": "Walters State Community College",
    "WEB_STA_SB": "Weber State University",
    "WES_CAR_SB": "Western Carolina University",
    "WES_NEB_SB": "Western Nebraska Community College",
    "WIN_EAG_SB": "Winthrop University",
    "WIS_BAD_SB": "University of Wisconsin",
    "WOF_TER_SB": "Wofford College",
    "YAL_BUL_SB": "Yale University",
    "DEL_STA_SB": "Delaware State University",
    "ALA_STA_SB": "Alabama State University",
    "DAY_FLY_SB": "University of Dayton",
    "MUR_STA1_SB": "Murray State University",
    "OAK_GOL_SB": "Oakland University",
    "TAR_STA_SB": "Tarleton State University",
    "UNI_PIT_SB": "University of Pittsburgh",
    "EAS_KEN_SB": "Eastern Kentucky University",
    "MOR_STA_SB": "Morehead State University",
    
}


# Put your Purdue logo image inside app/static/
# Example file name:
PURDUE_LOGO_FILENAME = "purdue-logo.png"
PURDUE_LOGO_SRC = PURDUE_LOGO_FILENAME  # served from static_assets

FIG_SIZE = (3.6, 2.8)

COLUMNS_TO_KEEP = [
    "PitchNo", "Date", "Time", "PitchofPA",
    "Pitcher", "PitcherId", "PitcherThrows", "PitcherTeam",
    "Batter", "BatterId", "BatterSide", "BatterTeam",
    "Balls", "Strikes", "PitchCall", "TaggedPitchType",
    "RelSpeed", "SpinRate", "SpinAxis", "Tilt",
    "InducedVertBreak", "HorzBreak",
    "PlateLocHeight", "PlateLocSide",
    "TaggedHitType", "PlayResult", "KorBB", "ExitSpeed", "Angle", "Direction",
]

PITCH_TYPE_COL = "TaggedPitchType"
X_MOV = "HorzBreak"
Y_MOV = "InducedVertBreak"

# Fixed strike zone (feet)
ZONE_LEFT, ZONE_RIGHT = -0.83, 0.83
ZONE_BOTTOM, ZONE_TOP = 1.5, 3.5

MOV_XLIM = (-20, 20)
MOV_YLIM = (-20, 20)

# Purdue detection (robust: does not require exact team string)

PURDUE_CODE = "PUR_BOI_SB"

def is_purdue_team(team_value: str) -> bool:
    if team_value is None or pd.isna(team_value):
        return False
    return str(team_value).strip() == PURDUE_CODE

# Pitch colors (consistent)
PITCH_TYPE_FIXED_COLORS = {
    "Fastball":  "#475569",
    "Changeup":  "#D4A017",
    "Curveball": "#1B9E77",
    "Riseball":  "#1F77FF",
    "Dropball":  "#D62728",
    "Screwball": "#9467BD",
    "Offspeed":  "#FF7F0E",
    "Riser": "#1F77FF",
    "Rise":  "#1F77FF",
    "Drop":  "#D62728",
}
PITCH_TYPE_FALLBACK_COLORS = ["#17BECF", "#8C564B", "#E377C2", "#7F7F7F"]


# ---------------------------------------------------------------------------
# Session type inference (Purdue only)
# ---------------------------------------------------------------------------
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

def infer_session_type_for_purdue(df, filename=""):
    if df is None or df.empty:
        out = df.copy()
        out["SessionType"] = "unknown"
        out["SessionTypeReason"] = "Empty file"
        return out

    out = df.copy()

    if "-bp-" in filename.lower():
        out["SessionType"] = "batting_practice"
        out["SessionTypeReason"] = "Filename contains -BP-"
        return out

    

    notes_col = "Notes" if "Notes" in out.columns else None
    batter_name_col = "Batter" if "Batter" in out.columns else None
    batter_id_col = "BatterId" if "BatterId" in out.columns else None
    pitcher_name_col = "Pitcher" if "Pitcher" in out.columns else None
    pitcher_id_col = "PitcherId" if "PitcherId" in out.columns else None
    batter_team_col = "BatterTeam" if "BatterTeam" in out.columns else None
    pitcher_team_col = "PitcherTeam" if "PitcherTeam" in out.columns else None

    notes_clean = (
        out[notes_col].fillna("").astype(str).str.strip().str.lower()
        if notes_col else pd.Series("", index=out.index)
    )

    batter_clean = (
        out[batter_name_col].apply(clean_player_name)
        if batter_name_col else pd.Series("", index=out.index)
    )

    batter_has_purdue = contains_purdue(out[batter_team_col]) if batter_team_col else False
    pitcher_has_purdue = contains_purdue(out[pitcher_team_col]) if pitcher_team_col else False
    batter_has_non_purdue = contains_non_purdue(out[batter_team_col]) if batter_team_col else False
    pitcher_has_non_purdue = contains_non_purdue(out[pitcher_team_col]) if pitcher_team_col else False

    batter_has_data = False
    if batter_name_col:
        batter_has_data = batter_has_data or series_has_data(out[batter_name_col])
    if batter_id_col:
        batter_has_data = batter_has_data or series_has_data(out[batter_id_col])

    pitcher_has_data = False
    if pitcher_name_col:
        pitcher_has_data = pitcher_has_data or series_has_data(out[pitcher_name_col])
    if pitcher_id_col:
        pitcher_has_data = pitcher_has_data or series_has_data(out[pitcher_id_col])

    pitcher_name_empty = True if not pitcher_name_col else series_all_empty(out[pitcher_name_col])
    pitcher_id_empty = True if not pitcher_id_col else series_all_empty(out[pitcher_id_col])

    batter_is_active = batter_clean.isin(ACTIVE_ROSTER_2026).any()
    batter_is_former = (
        batter_clean.isin(KNOWN_FORMER_PLAYERS)
        & ~batter_clean.isin(ACTIVE_ROSTER_2026)
    ).any()

    # 1. Bullpen
    bullpen_by_notes = notes_clean.str.contains("bullpen", na=False).any()
    bullpen_by_former_player = pitcher_has_purdue and batter_is_former

    if bullpen_by_notes and pitcher_has_purdue:
        out["SessionType"] = "bullpen"
        out["SessionTypeReason"] = "Notes says bullpen"
        return out

    if bullpen_by_former_player:
        out["SessionType"] = "bullpen"
        out["SessionTypeReason"] = "Known former player batter with Purdue pitcher"
        return out

    # 2. Batting Practice
    if batter_has_purdue and batter_has_data and pitcher_name_empty and pitcher_id_empty:
        out["SessionType"] = "batting_practice"
        out["SessionTypeReason"] = "Purdue batter only; no pitcher name or pitcher ID"
        return out

    # 3. Live
    if (
        (batter_has_purdue and pitcher_has_non_purdue) or
        (batter_has_non_purdue and pitcher_has_purdue) or
        (batter_has_non_purdue and pitcher_has_non_purdue)
    ):
        out["SessionType"] = "live"
        out["SessionTypeReason"] = "External or non-Purdue matchup"
        return out

    # 4. Scrimmage
    if batter_has_purdue and pitcher_has_purdue and pitcher_has_data and batter_has_data:
        out["SessionType"] = "scrimmage"
        out["SessionTypeReason"] = "Purdue vs Purdue"
        return out

    # 5. Unknown
    out["SessionType"] = "unknown"
    out["SessionTypeReason"] = "Could not classify"
    return out


def apply_session_filter_for_team(df, team, session_value):
    if df is None or df.empty:
        return df
    if session_value == "all":
        return df
    if not is_purdue_team(team) and session_value in {"bullpen", "batting_practice", "scrimmage"}:
        return df
    return df[df["SessionType"] == session_value]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_csv_paths():
    if not os.path.isdir(V3_PATH):
        return []
    pattern = os.path.join(V3_PATH, "**", "*.csv")
    paths = sorted(glob.glob(pattern, recursive=True))
    out = []
    for p in paths:
        try:
            rel = os.path.relpath(p, V3_PATH)
        except ValueError:
            rel = os.path.basename(p)
        out.append((rel, p))
    return out

def _data_dir_mtime():
    if not os.path.isdir(V3_PATH):
        return 0
    try:
        t = os.path.getmtime(V3_PATH)
        for e in os.listdir(V3_PATH):
            p = os.path.join(V3_PATH, e)
            t = max(t, os.path.getmtime(p))
        return t
    except Exception:
        return 0

def get_csv_paths_with_dates():
    cache_path = os.path.join(APP_DIR, ".csv_metadata_cache.json")
    v3_mtime = _data_dir_mtime()

    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            if data.get("v3_path") == V3_PATH and data.get("mtime") == v3_mtime:
                rows = []
                gmin_s, gmax_s = data.get("global_min"), data.get("global_max")
                gmin = date.fromisoformat(gmin_s) if gmin_s else None
                gmax = date.fromisoformat(gmax_s) if gmax_s else None
                for rel, full, dmin_s, dmax_s in data.get("rows", []):
                    dmin = date.fromisoformat(dmin_s) if dmin_s else None
                    dmax = date.fromisoformat(dmax_s) if dmax_s else None
                    if dmin and dmax:
                        rows.append((rel, full, dmin, dmax))
                return rows, gmin, gmax
        except Exception:
            pass

    rows = []
    gmin, gmax = None, None
    for rel, full in get_csv_paths():
        try:
            df = pd.read_csv(full, usecols=["Date"])
        except Exception:
            continue
        if "Date" not in df.columns or df["Date"].empty:
            continue
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if dates.empty:
            continue
        dmin = dates.min().date()
        dmax = dates.max().date()
        rows.append((rel, full, dmin, dmax))
        gmin = dmin if (gmin is None or dmin < gmin) else gmin
        gmax = dmax if (gmax is None or dmax > gmax) else gmax

    try:
        cache_data = {
            "v3_path": V3_PATH,
            "mtime": v3_mtime,
            "rows": [[rel, full, str(dmin), str(dmax)] for rel, full, dmin, dmax in rows],
            "global_min": str(gmin) if gmin else None,
            "global_max": str(gmax) if gmax else None,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)
    except Exception:
        pass

    return rows, gmin, gmax

# Map Purdue batter names → correct BatterIds (fix for Trackman placeholder IDs)
PURDUE_BATTER_ID_MAP = {
    "Armstrong, Ansley": "100000001217",
    "Bailey, Emma": "100000001224",
    "Bailey, Kyndall": "100000001233",
    "Banks, Khloe": "100000001231",
    "Campbell, Ashlynn": "100000001219",
    "Condon, Maura": "100000001235",
    "Douglas, Bella": "1000000001384",
    "Fontenot, Bri": "1000000001385",
    "Franks, Kylie": "100000001232",
    "Gossett, Julia": "100000001228",
    "Klochack, Kendall": "100000001230",
    "Krantz, Jensen": "100000001225",
    "McFadden, Olivia": "100000001237",
    "Meeks, Alivia": "100000001216",
    "Moore, Anna": "1000000001383",
    "Moore, Malone": "100000002641",
    "Painter, Haley": "100000005709",
    "Perez, Brooke": "1000000001389",
    "Polar, Moriah": "100000001236",
    "Rainey, Kendyl": "1000000001382",
    "Reefe, Delaney": "100000001223",
    "Sarago, Kate": "1000000001388",
    "Sosa, Gabby": "1000000001386",
    "Waggoner, Haley": "1000000001387",
}


def load_and_clean_csv(full_path: str) -> pd.DataFrame | None:
    try:
        # Force ID columns to string so placeholder IDs don't get coerced to float
        df = pd.read_csv(full_path, dtype={"BatterId": str, "PitcherId": str})
    except Exception:
        try:
            df = pd.read_csv(full_path)
        except Exception:
            return None

    cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    if not cols:
        return None
    df = df[cols].copy()

    for c in [
        "Pitcher", "Batter", PITCH_TYPE_COL, "PitchCall",
        "PitcherTeam", "BatterTeam", "PitcherThrows", "BatterSide"
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Fix batter IDs: replace placeholder/scientific-notation IDs
    if "BatterTeam" in df.columns and "Batter" in df.columns and "BatterId" in df.columns:
        df["BatterId"] = df["BatterId"].astype(str).str.strip()

        def _is_placeholder(bid):
            s = str(bid).strip().lower()
            # Scientific notation like '1e+11', '1e+12', or the expanded form
            if "e+" in s or "e-" in s:
                return True
            if s in ("", "nan", "none", "1000000000000", "100000000000"):
                return True
            return False

        # For Purdue batters: use name → correct ID mapping
        is_purdue = df["BatterTeam"].astype(str).str.strip() == PURDUE_CODE
        if is_purdue.any():
            def _fix_purdue_bid(row):
                if _is_placeholder(row["BatterId"]):
                    name = str(row["Batter"]).strip()
                    return PURDUE_BATTER_ID_MAP.get(name, f"SYN_{name}")
                return row["BatterId"]
            df.loc[is_purdue, "BatterId"] = df.loc[is_purdue].apply(_fix_purdue_bid, axis=1)

        # For non-Purdue batters with placeholder IDs: synthesize from name+team
        # so each unique batter gets their own ID instead of all collapsing to one
        non_purdue = ~is_purdue
        if non_purdue.any():
            def _fix_other_bid(row):
                if _is_placeholder(row["BatterId"]):
                    name = str(row["Batter"]).strip()
                    team = str(row["BatterTeam"]).strip()
                    return f"SYN_{team}_{name}"
                return row["BatterId"]
            df.loc[non_purdue, "BatterId"] = df.loc[non_purdue].apply(_fix_other_bid, axis=1)

    return df

# ── Rapsodo CSV loader ────────────────────────────────────────────────────
RAPSODO_PITCH_MAP = {
    "Fastball": "Fastball",
    "Riser": "Riseball",
    "CurveBall": "Curveball",
    "ChangeUp": "Changeup",
    "Dropball": "Dropball",
    "OffSpeedDrop": "Offspeed",
    "OffSpeedRise": "Offspeed",
    "TwoSeamFastball": "Fastball",
    "DropCurve": "Drop-Curve",
}

RAPSODO_COL_MAP = {
    "No": "PitchNo",
    "Pitch Type": "TaggedPitchType",
    "Velocity": "RelSpeed",
    "Total Spin": "SpinRate",
    "Strike Zone Side": "PlateLocSide",
    "Strike Zone Height": "PlateLocHeight",
    "HB (trajectory)": "HorzBreak",
    "VB (trajectory)": "InducedVertBreak",
    "Spin Direction": "SpinAxis",
    "Release Height": "RelHeight",
    "Release Side": "RelSide",
    "Release Extension (ft)": "Extension",
}

# Map Rapsodo Player IDs → Trackman PitcherIds
RAPSODO_TO_TRACKMAN_ID = {
    "1065934": "100000001224",    # Emma Bailey
    "1502155": "1000000001385",   # Bri Fontenot
    "907287":  "1000000001389",   # Brooke Perez
    "1120697": "100000001228",    # Julia Gossett
    "1500934": "100000002641",    # Malone Moore
}

def load_rapsodo_csv(full_path: str) -> pd.DataFrame | None:
    try:
        # Find header row (has "No" and "Date")
        with open(full_path) as fh:
            lines = fh.readlines()
        header_row = None
        player_name = ""
        player_id = ""
        for i, line in enumerate(lines):
            if '"Player Name:"' in line or 'Player Name:' in line:
                player_name = line.split(",", 1)[1].strip().strip('"')
            if '"Player ID:"' in line or 'Player ID:' in line:
                player_id = line.split(",", 1)[1].strip().strip('"')
            if '"No"' in line and '"Date"' in line:
                header_row = i
                break
        if header_row is None:
            return None

        df = pd.read_csv(full_path, skiprows=header_row)
    except Exception:
        return None

    if df.empty:
        return None

    # Rename columns to Trackman names
    df = df.rename(columns=RAPSODO_COL_MAP)

    # Map pitch type names
    if "TaggedPitchType" in df.columns:
        df["TaggedPitchType"] = df["TaggedPitchType"].astype(str).str.strip().map(
            lambda pt: RAPSODO_PITCH_MAP.get(pt, pt)
        )

    # Map Is Strike → PitchCall
    if "Is Strike" in df.columns:
        df["PitchCall"] = df["Is Strike"].astype(str).str.strip().map(
            {"Y": "StrikeCalled", "N": "BallCalled"}
        ).fillna("BallCalled")

    # Ensure numeric columns are numeric
    for c in ["RelSpeed", "SpinRate", "PlateLocSide", "PlateLocHeight",
              "HorzBreak", "InducedVertBreak"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rapsodo reports location in inches — convert to feet for Trackman compatibility
    for c in ["PlateLocSide", "PlateLocHeight"]:
        if c in df.columns:
            df[c] = df[c] / 12.0

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")

    # Add pitcher info — map to Trackman PitcherId for consistency
    # Convert "Emma Bailey" → "Bailey, Emma" to match Trackman format
    name_parts = player_name.strip().split()
    if len(name_parts) >= 2:
        pitcher_name = f"{name_parts[-1]}, {' '.join(name_parts[:-1])}"
    else:
        pitcher_name = player_name
    df["Pitcher"] = pitcher_name
    trackman_id = RAPSODO_TO_TRACKMAN_ID.get(str(player_id).strip(), str(player_id))
    df["PitcherId"] = trackman_id
    df["PitcherTeam"] = PURDUE_CODE
    df["PitcherThrows"] = "Right"

    # Mark as Rapsodo data
    df["DataSource"] = "rapsodo"

    # Keep only columns the app uses (plus extras)
    keep = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    extra = ["DataSource", "RelHeight", "RelSide", "Extension"]
    keep += [c for c in extra if c in df.columns and c not in keep]
    df = df[keep].copy()

    return df


# ---------------------------------------------------------------------------
# HitTrax data (session-aggregated batter data)
# ---------------------------------------------------------------------------
# Map HitTrax filename stem → Trackman-format player name
HITTRAX_NAME_MAP = {
    "Kloe_Banks":       "Banks, Khloe",
    "Anna_Moore":       "Moore, Anna",
    "Ansley_Amstrong":  "Armstrong, Ansley",
    "Bella_Douglas":    "Douglas, Bella",
    "Delaney_Reefe":    "Reefe, Delaney",
    "Gabby_Sosa":       "Sosa, Gabby",
    "Haley_Painter":    "Painter, Haley",
    "Haley_Waggoner":   "Waggoner, Haley",
    "Jensen_Krantz":    "Krantz, Jensen",
    "Jordyn_Rudd-Lee":  "Rudd-Lee, Jordyn",
    "Julia_Gossett":    "Gossett, Julia",
    "Kate_Sarago":      "Sarago, Kate",
    "Kendyl_Rainey":    "Rainey, Kendyl",
    "Kylie_Franks":     "Franks, Kylie",
    "Maura_Condon":     "Condon, Maura",
    "Moriah_Polar":     "Polar, Moriah",
    "Alivia_Meeks":     "Meeks, Alivia",
}

def load_hittrax_csv(full_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(full_path)
    except Exception:
        return None
    if df.empty:
        return None

    # Strip leading/trailing whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Parse date
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df[df["Date"].notna()].copy()
    if df.empty:
        return None

    # Coerce numeric columns
    numeric_cols = ["AB", "H", "EBH", "HR", "Points", "LD %", "FB %", "GB %",
                    "AvgV", "MaxV", "Dist"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Parse .XXX rate columns as floats (HHA, AVG, SLG, LPH)
    for c in ["HHA", "AVG", "SLG", "LPH"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Player name from filename stem
    stem = os.path.splitext(os.path.basename(full_path))[0]
    player_name = HITTRAX_NAME_MAP.get(stem, stem.replace("_", ", "))
    df["Player"] = player_name
    df["BatterTeam"] = PURDUE_CODE
    df["DataSource"] = "hittrax"
    df["DateOnly"] = df["Date"].dt.date

    return df


def build_hittrax_df():
    hittrax_dir = os.path.join(APP_DIR, "data", "hittrax")
    if not os.path.isdir(hittrax_dir):
        return pd.DataFrame()
    dfs = []
    for fname in sorted(glob.glob(os.path.join(hittrax_dir, "*.csv"))):
        df = load_hittrax_csv(fname)
        if df is None or df.empty:
            continue
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


HITTRAX_DF = build_hittrax_df()


ACTIVE_ROSTER_2026 = {
    "Rainey, Kendyl",
    "Banks, Khloe",
    "Franks, Kylie",
    "Moore, Anna",
    "Reefe, Delaney",
    "Douglas, Bella",
    "Krantz, Jensen",
    "Condon, Maura",
    "Fontenot, Bri",
    "Moore, Malone",
    "Campbell, Ashlynn",
    "Bailey, Emma",
    "Sosa, Gabby",
    "Waggoner, Haley",
    "Gossett, Julia",
    "Sarago, Kate",
    "Perez, Brooke",
    "Armstrong, Ansley",
    "Painter, Haley",
    "Polar, Moriah",
}

KNOWN_FORMER_PLAYERS = {
    "Klochack, Kendall",
    "McFadden, Olivia",
    "Bailey, Kyndall",
}

PURDUE_TEAM_KEYS = {"PUR_BOI_SB"}

def clean_player_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = " ".join(s.split())
    parts = [p.strip().title() for p in s.split(",")]
    return ", ".join(parts)

def series_has_data(s):
    if s is None or len(s) == 0:
        return False
    return s.fillna("").astype(str).str.strip().ne("").any()

def series_all_empty(s):
    if s is None or len(s) == 0:
        return True
    return s.fillna("").astype(str).str.strip().eq("").all()

def contains_purdue(series):
    if series is None or len(series) == 0:
        return False
    vals = set(series.fillna("").astype(str).str.strip().str.upper())
    return any(v in PURDUE_TEAM_KEYS for v in vals if v)

def contains_non_purdue(series):
    if series is None or len(series) == 0:
        return False
    vals = set(series.fillna("").astype(str).str.strip().str.upper())
    vals = {v for v in vals if v}
    return any(v not in PURDUE_TEAM_KEYS for v in vals)

def build_pitch_color_map(pitch_types):
    out = {}
    unknown = sorted([p for p in pitch_types if pd.notna(p) and p not in PITCH_TYPE_FIXED_COLORS])
    unknown_idx = {pt: i for i, pt in enumerate(unknown)}   # ← build O(1) lookup once
    for pt in pitch_types:
        if pd.isna(pt):
            continue
        if pt in PITCH_TYPE_FIXED_COLORS:
            out[pt] = PITCH_TYPE_FIXED_COLORS[pt]
        else:
            idx = unknown_idx.get(pt, 0)
            out[pt] = PITCH_TYPE_FALLBACK_COLORS[idx % len(PITCH_TYPE_FALLBACK_COLORS)]
    return out

def pitch_alpha(pt: str, selected: str) -> float:
    if not selected:
        return 0.7
    if pt == selected:
        return 1.0
    return 0.15


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


def is_valid_pitch_type(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return (
        series.notna()
        & s.ne("")
        & s.ne("undefined")
        & s.ne("other")
        & s.ne("nan")
    )

def compute_batter_stats(df: pd.DataFrame, batter_id) -> dict:
    empty = dict(PA=0, AB=0, H=0, doubles=0, triples=0, HR=0,
                 BB=0, K=0, HBP=0, BA=None, OBP=None, SLG=None,
                 OPS=None, wOBA=None)
    if df is None or df.empty or batter_id is None:
        return empty

    d = df[df["BatterId"].astype(str) == str(batter_id)].copy()
    if d.empty:
        return empty

    pr = d["PlayResult"].astype(str).str.strip() if "PlayResult" in d.columns else pd.Series("", index=d.index)
    pc = d["PitchCall"].astype(str).str.strip()  if "PitchCall"  in d.columns else pd.Series("", index=d.index)

    # ── PA: count using PitchofPA == 1 (start of each new PA) ──────────────
    if "PitchofPA" in d.columns:
        PA = int((pd.to_numeric(d["PitchofPA"], errors="coerce") == 1).sum())
    else:
        # fallback: count terminal pitches
        TERMINAL = {"Single","Double","Triple","HomeRun","Out","FieldersChoice",
                    "Error","Walk","Strikeout","HitByPitch","SacrificeFly",
                    "SacrificeBunt","CatcherInterference"}
        PA = int((pr.isin(TERMINAL) | pc.eq("HitByPitch")).sum())

    if PA == 0:
        return empty

    # ── Use terminal pitch rows for hit/walk/K counting ────────────────────
    TERMINAL = {"Single","Double","Triple","HomeRun","Out","FieldersChoice",
                "Error","Walk","Strikeout","HitByPitch","SacrificeFly",
                "SacrificeBunt","CatcherInterference"}
    t = d[pr.isin(TERMINAL) | pc.eq("HitByPitch")].copy()

    if t.empty:
        # PA exists but no PlayResult data — return PA only
        return dict(PA=PA, AB=0, H=0, doubles=0, triples=0, HR=0,
                    BB=0, K=0, HBP=0, BA=None, OBP=None, SLG=None,
                    OPS=None, wOBA=None)

    t_pr = t["PlayResult"].astype(str).str.strip() if "PlayResult" in t.columns else pd.Series("", index=t.index)
    t_pc = t["PitchCall"].astype(str).str.strip()  if "PitchCall"  in t.columns else pd.Series("", index=t.index)

    singles = int(t_pr.eq("Single").sum())
    doubles = int(t_pr.eq("Double").sum())
    triples = int(t_pr.eq("Triple").sum())
    HR      = int(t_pr.eq("HomeRun").sum())
    H       = singles + doubles + triples + HR
    BB      = int(t_pr.eq("Walk").sum())
    K       = int(t_pr.eq("Strikeout").sum())
    HBP     = int((t_pr.eq("HitByPitch") | t_pc.eq("HitByPitch")).sum())
    AB      = max(PA - BB - HBP, 0)
    TB      = singles + 2*doubles + 3*triples + 4*HR

    BA   = round(H / AB, 3)                    if AB  > 0 else None
    OBP  = round((H + BB + HBP) / PA, 3)       if PA  > 0 else None
    SLG  = round(TB / AB, 3)                   if AB  > 0 else None
    OPS  = round(OBP + SLG, 3)                 if (OBP and SLG) else None
    wOBA = round((0.690*BB + 0.720*HBP + 0.880*singles +
                  1.242*doubles + 1.569*triples + 2.007*HR) / PA, 3) if PA > 0 else None

    return dict(PA=PA, AB=AB, H=H, doubles=doubles, triples=triples, HR=HR,
                BB=BB, K=K, HBP=HBP, BA=BA, OBP=OBP, SLG=SLG, OPS=OPS, wOBA=wOBA)

def compute_usage(df: pd.DataFrame, pitcher_id) -> pd.DataFrame:
    if df is None or df.empty or pitcher_id is None:
        return pd.DataFrame(columns=[PITCH_TYPE_COL, "pitch_count", "usage_pct"])

    mask = (
        (df["PitcherId"].astype(str) == str(pitcher_id)) &
        is_valid_pitch_type(df[PITCH_TYPE_COL])
    )
    d = df[mask].copy()

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

def filter_df_to_pitch_type(df: pd.DataFrame, pitch_type_value: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if pitch_type_value in (None, "", "all"):
        return df
    return df[df[PITCH_TYPE_COL].astype(str).str.strip() == str(pitch_type_value)].copy()


def compute_pitch_metrics(df: pd.DataFrame, pitcher_id):
    if df is None or df.empty or pitcher_id is None:
        return pd.DataFrame()

    required = {
        "PitcherId", PITCH_TYPE_COL, "PitchNo", "RelSpeed",
        "PitchCall", "PlateLocSide", "PlateLocHeight"
    }
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    mask = (
        (df["PitcherId"].astype(str) == str(pitcher_id)) &
        is_valid_pitch_type(df[PITCH_TYPE_COL])
    )
    d = df[mask].copy()

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
            ivb_avg=("InducedVertBreak", "mean"),
            hb_avg=("HorzBreak", "mean"),
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

def compute_comparison_metrics(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "pitch_count": 0,
            "avg_velo": np.nan,
            "max_velo": np.nan,
            "velo_sd": np.nan,
            "spin_rate": np.nan,
            "ivb_avg": np.nan,
            "hb_avg": np.nan,
            "strike_pct": np.nan,
            "whiff_pct": np.nan,
            "swing_pct": np.nan,
        }

    pitch_count = len(df)

    velo = pd.to_numeric(df["RelSpeed"], errors="coerce") if "RelSpeed" in df.columns else pd.Series(dtype=float)
    spin = pd.to_numeric(df["SpinRate"], errors="coerce") if "SpinRate" in df.columns else pd.Series(dtype=float)
    velo_sd = velo.std()
    ivb = pd.to_numeric(df["InducedVertBreak"], errors="coerce") if "InducedVertBreak" in df.columns else pd.Series(dtype=float)
    hb  = pd.to_numeric(df["HorzBreak"],        errors="coerce") if "HorzBreak"        in df.columns else pd.Series(dtype=float)
    pc = df["PitchCall"].astype(str).str.strip() if "PitchCall" in df.columns else pd.Series("", index=df.index)
    is_strike = pc.isin({"StrikeCalled", "StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"})
    is_swing  = pc.isin({"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"})
    is_whiff  = pc.eq("StrikeSwinging")

    return {
        "pitch_count": pitch_count,
        "avg_velo":    velo.mean(),
        "max_velo":    velo.max(),
        "velo_sd":  velo_sd,
        "spin_rate":   spin.mean(),
        "ivb_avg":     ivb.mean(),  
        "hb_avg":      hb.mean(),
        "strike_pct":  is_strike.mean() if pitch_count > 0 else np.nan,
        "swing_pct":   is_swing.mean()  if pitch_count > 0 else np.nan,
        "whiff_pct":   (is_whiff.sum() / is_swing.sum()) if is_swing.sum() > 0 else np.nan,
    }

# ---------------------------------------------------------------------------
# Prediction tab: descriptive + ML pitch-type analysis
# ---------------------------------------------------------------------------
PREDICTION_HARD_CONTACT_EV = 85.0
PREDICTION_MIN_STRIKE_N = 12
PREDICTION_MIN_SWINGS_PUTAWAY = 8
PREDICTION_MIN_CONTACT_CAUTION = 5

def _prediction_sample_warning(n: int) -> str:
    if n < 6:   return "Very low sample — interpret cautiously"
    if n < 12:  return "Low sample"
    if n < 25:  return "Moderate sample"
    return ""


def build_prediction_by_pitch_type(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "PitchCall" not in df.columns or PITCH_TYPE_COL not in df.columns:
        return pd.DataFrame()
    d = df[is_valid_pitch_type(df[PITCH_TYPE_COL])].copy()
    if d.empty:
        return pd.DataFrame()

    swing_events = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
    strike_events = {"StrikeCalled", "StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
    contact_events = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}

    rows = []
    for ptype, g in d.groupby(PITCH_TYPE_COL, dropna=False):
        pc = g["PitchCall"].astype(str).str.strip()
        n = len(g)
        is_swing = pc.isin(swing_events)
        is_whiff = pc.eq("StrikeSwinging")
        is_contact = pc.isin(contact_events)
        in_play = pc.eq("InPlay")
        ev_s = pd.to_numeric(g["ExitSpeed"], errors="coerce") if "ExitSpeed" in g.columns else pd.Series(np.nan, index=g.index)

        swing_count = int(is_swing.sum())
        whiff_count = int(is_whiff.sum())
        contact_count = int(is_contact.sum())
        hard_contact_count = int((in_play & (ev_s >= PREDICTION_HARD_CONTACT_EV)).sum())

        rows.append({
            PITCH_TYPE_COL: ptype,
            "pitch_count": n,
            "strike_pct": float(pc.isin(strike_events).mean()) if n else np.nan,
            "contact_pct": float(is_contact.mean()) if n else np.nan,
            "swing_count": swing_count,
            "whiff_count": whiff_count,
            "contact_count": contact_count,
            "whiff_pct": (whiff_count / swing_count) if swing_count > 0 else np.nan,
            "hard_contact_count": hard_contact_count,
            "hard_contact_risk": (hard_contact_count / contact_count) if contact_count > 0 else np.nan,
            "sample_warning": _prediction_sample_warning(n),
        })

    out = pd.DataFrame(rows)
    return out.sort_values("pitch_count", ascending=False).reset_index(drop=True) if not out.empty else out


def select_prediction_summary(pred: pd.DataFrame) -> dict:
    out = {"best_strike": None, "best_putaway": None, "caution": None}
    if pred is None or pred.empty or PITCH_TYPE_COL not in pred.columns:
        return out
    p = pred.copy()
    pcol = PITCH_TYPE_COL

    def _build(row):
        return {
            "pitch": str(row[pcol]),
            "strike_pct": float(row["strike_pct"]) if pd.notna(row["strike_pct"]) else np.nan,
            "whiff_pct": float(row["whiff_pct"]) if pd.notna(row["whiff_pct"]) else np.nan,
            "contact_pct": float(row["contact_pct"]) if pd.notna(row["contact_pct"]) else np.nan,
            "hard_contact_risk": float(row["hard_contact_risk"]) if pd.notna(row["hard_contact_risk"]) else np.nan,
            "pitch_count": int(row["pitch_count"]),
            "swing_count": int(row["swing_count"]),
            "contact_count": int(row["contact_count"]),
            "hard_contact_count": int(row["hard_contact_count"]),
            "sample_warning": str(row["sample_warning"]),
        }

    # Best strike
    tier = p[p["pitch_count"] >= PREDICTION_MIN_STRIKE_N]
    if tier.empty: tier = p[p["pitch_count"] >= 6]
    if tier.empty: tier = p
    if not tier.empty:
        best = tier.loc[tier["strike_pct"].idxmax()]
        out["best_strike"] = {**_build(best), "coach_blurb": "Highest strike rate — a solid default when you need a strike."}

    # Best putaway
    tier = p[p["swing_count"] >= PREDICTION_MIN_SWINGS_PUTAWAY]
    if tier.empty: tier = p[p["swing_count"] >= 4]
    if tier.empty: tier = p
    tier = tier[tier["swing_count"] > 0]
    if not tier.empty:
        tier = tier.assign(_wp=tier["whiff_pct"].fillna(0.0))
        best = tier.loc[tier["_wp"].idxmax()]
        out["best_putaway"] = {**_build(best), "coach_blurb": "Strongest whiff rate — lean on it in two-strike situations."}

    # Caution
    tier = p[p["contact_count"] >= PREDICTION_MIN_CONTACT_CAUTION]
    if tier.empty: tier = p[p["contact_count"] >= 2]
    tier = tier[tier["contact_count"] > 0]
    tier = tier[pd.notna(tier["hard_contact_risk"])]
    if not tier.empty and tier["hard_contact_count"].sum() > 0:
        best = tier.loc[tier["hard_contact_risk"].idxmax()]
        out["caution"] = {**_build(best), "coach_blurb": f"High share of hard contact (EV ≥ {int(PREDICTION_HARD_CONTACT_EV)} mph) — be selective with location."}
    else:
        out["caution"] = {"pitch": "—", "coach_blurb": "Not enough hard contact data to flag a caution pitch."}

    return out


def format_prediction_table_display(pred: pd.DataFrame, summary: dict) -> pd.DataFrame:
    if pred is None or pred.empty:
        return pd.DataFrame(columns=["Pitch", "Strike %", "Whiff %", "Contact %", "Hard Contact Risk", "Sample", "Recommendation"])
    pcol = PITCH_TYPE_COL
    bs = (summary.get("best_strike") or {}).get("pitch")
    bp = (summary.get("best_putaway") or {}).get("pitch")
    ca = (summary.get("caution") or {}).get("pitch")
    recs = []
    for _, row in pred.iterrows():
        name = str(row[pcol])
        tags = []
        if bs and name == bs: tags.append("Best strike")
        if bp and name == bp: tags.append("Best put-away")
        if ca and name == ca and ca != "—": tags.append("Caution")
        recs.append("; ".join(tags) if tags else "—")
    return pd.DataFrame({
        "Pitch": pred[pcol].astype(str),
        "Strike %": pred["strike_pct"].map(lambda x: format_pct(x) if pd.notna(x) else "—"),
        "Whiff %": pred["whiff_pct"].map(lambda x: format_pct(x) if pd.notna(x) else "—"),
        "Contact %": pred["contact_pct"].map(lambda x: format_pct(x) if pd.notna(x) else "—"),
        "Hard Contact Risk": pred["hard_contact_risk"].map(lambda x: format_pct(x) if pd.notna(x) else "—"),
        "Sample": pred["pitch_count"].astype(int),
        "Sample Note": pred["sample_warning"].map(lambda s: s if str(s).strip() else "—"),
        "Recommendation": recs,
    })


def get_pitcher_team_logo_text(team_code: str) -> str:
    if team_code == PURDUE_CODE:
        return "P"
    return "O"

# ---------------------------------------------------------------------------
# Load csv metadata once
# ---------------------------------------------------------------------------
csv_paths_with_dates, global_date_min, global_date_max = get_csv_paths_with_dates()

def build_master_df():
    dfs = []

    # Load Trackman data
    for rel, full, dmin, dmax in csv_paths_with_dates:
        df = load_and_clean_csv(full)
        if df is None or df.empty or "Date" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].notna()].copy()
        if df.empty:
            continue

        df["DateOnly"] = df["Date"].dt.date
        df["DataSource"] = "trackman"
        df = infer_session_type_for_purdue(df, filename=rel)

        dfs.append(df)

    # Load Rapsodo data
    rapsodo_dir = os.path.join(APP_DIR, "data", "rapsodo")
    if os.path.isdir(rapsodo_dir):
        for fname in glob.glob(os.path.join(rapsodo_dir, "*.csv")):
            df = load_rapsodo_csv(fname)
            if df is None or df.empty or "Date" not in df.columns:
                continue
            df = df[df["Date"].notna()].copy()
            if df.empty:
                continue
            df["DateOnly"] = df["Date"].dt.date
            df["SessionType"] = "bullpen"
            df["SessionTypeReason"] = "Rapsodo bullpen data"
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    master = pd.concat(dfs, ignore_index=True)

    # Exclude Trackman test/demo data
    EXCLUDE_TEAMS = {"TRA_TRA_SB", "TRA_TRA1_SB"}
    for col in ["PitcherTeam", "BatterTeam"]:
        if col in master.columns:
            master = master[~master[col].astype(str).str.strip().isin(EXCLUDE_TEAMS)]

    return master

MASTER_DF = build_master_df()

# Precompute per-source slices once at startup so current_df() doesn't have to
# re-filter MASTER_DF by DataSource on every reactive invalidation.
def _build_source_slices(master):
    if master is None or master.empty or "DataSource" not in master.columns:
        return {"trackman": master, "rapsodo": master, "collective": master}
    return {
        "trackman": master[master["DataSource"] == "trackman"].reset_index(drop=True),
        "rapsodo":  master[master["DataSource"] == "rapsodo"].reset_index(drop=True),
        "collective": master,  # no filter
    }
SOURCE_SLICES = _build_source_slices(MASTER_DF)

DEFAULT_SEASON = "spring_2026"

SEASON_DATE_MAP = {
    "spring_2026": (date(2026, 1, 1), date(2026, 6, 30)),
}

def clamp_date_range(start, end, global_min, global_max):
    if global_min is None or global_max is None:
        return None, None

    start = max(start, global_min)
    end = min(end, global_max)

    if start > end:
        return None, None

    return start, end

def get_initial_date_range(default_season):
    if global_date_min is None or global_date_max is None:
        return None, None

    if default_season == "all":
        return global_date_min, global_date_max

    season_start, season_end = SEASON_DATE_MAP.get(
        default_season, (global_date_min, global_date_max)
    )
    start, end = clamp_date_range(season_start, season_end, global_date_min, global_date_max)

    if start is None or end is None:
        return None, None

    return start, end

_date_start_value, _date_end_value = get_initial_date_range(DEFAULT_SEASON)
INITIAL_SEASON_SELECTION = DEFAULT_SEASON
if _date_start_value is None or _date_end_value is None:
    # If the default season has no overlap with loaded CSV dates,
    # fall back to "all" so date inputs remain interactive.
    INITIAL_SEASON_SELECTION = "all"
    _date_start_value, _date_end_value = get_initial_date_range("all")


# ---------------------------------------------------------------------------
# UI (Sketch Layout + Purdue Header)
# ---------------------------------------------------------------------------
app_ui = ui.page_fluid(
    # Preload Plotly.js so charts render on first load (not after a re-render)
    ui.tags.head(
        ui.tags.script(src="https://cdn.plot.ly/plotly-2.35.2.min.js"),
    ),
    ui.tags.style("""
        html, body {
            width: 100%;
            margin: 0;
            padding: 0;
            overflow-x: hidden;
            font-family: Arial, sans-serif;
            background-color: #f3f3f3;
        }

        .container-fluid {
            width: 100% !important;
            max-width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }

        .top-header {
            width: 100%;
            margin: 0;
            min-width: 100%;
            background-color: #000000;
            border-bottom: 6px solid #DDB945;
            box-sizing: border-box;
        }

        .sidebar .shiny-input-container,
        .sidebar .filter-title {
            font-weight: 900 !important;
        }

        .sidebar .shiny-input-container .form-check label.form-check-label {
            font-weight: 400 !important;
        }

        .top-header {
            width: 100%;
            margin: 0;
            min-width: 100%;
            background-color: #000000;
            border-bottom: 6px solid #DDB945;
            box-sizing: border-box;
        }

        .header-inner {
            width: 100%;
            display: grid;
            grid-template-columns: 260px 1fr;
            align-items: center;
            padding: 8px 20px;
            box-sizing: border-box;
        }

        .header-left {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            padding-left: 18px;
            box-sizing: border-box;
        }

        .header-left img {
            height: 40px;
            width: auto;
            display: block;
        }

        .header-title {
            text-align: center;
            color: #DDB945;
            font-size: 46px;
            font-weight: 900;
            letter-spacing: 0.5px;
            box-sizing: border-box;
        }

        .header-right {
            display: none;
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
            overflow-x: auto;
        }

        .tabs-wrap .nav-tabs {
            border-bottom: 1px solid #d6d6d6;
            background: #ffffff;
            padding: 8px 10px 0 10px;
            border-radius: 10px 10px 0 0;
        }

        .tabs-wrap .nav-tabs .nav-link {
            border: none;
            color: #111111;
            font-weight: 900;
            padding: 10px 18px;
            margin-right: 10px;
            background: transparent;
        }

        .tabs-wrap .nav-tabs .nav-link.active {
            color: #111111;
            position: relative;
        }

        .tabs-wrap .nav-tabs .nav-link.active::after {
            content: "";
            position: absolute;
            left: 12px;
            right: 12px;
            bottom: -1px;
            height: 3px;
            background: #DDB945;
        }

        .panel {
            background: #ffffff;
            border: 1px solid #d6d6d6;
            border-top: none;
            border-radius: 0 0 10px 10px;
            padding: 16px 16px 18px 16px;
            min-height: calc(100vh - 140px);
            overflow-y: auto;
            overflow-x: visible;
            height: auto;
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
            overflow: visible;
        }

        .card-fixed {
            min-height: 420px;
        }

        .legend-row{
            display:flex;
            flex-wrap:wrap;
            gap:14px;
            justify-content:center;
            align-items:center;
            margin: 8px 0 14px 0;
            padding: 8px 14px;
            border: 1px solid rgba(0,0,0,0.15);
            border-radius: 8px;
            background: #fff;
            font-size: 13px;
        }

        /* Date range labelless */
        .date-range-row label {
            display: none !important;
        }

        /* Chart expand modal */
        .expand-btn {
            position: absolute;
            top: 8px;
            right: 10px;
            cursor: pointer;
            font-size: 16px;
            color: #999;
            padding: 2px 5px;
            border-radius: 4px;
            transition: color 0.15s, background 0.15s;
            z-index: 10;
            line-height: 1;
        }
        .expand-btn:hover {
            color: #333;
            background: rgba(0,0,0,0.06);
        }
        .chart-modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 99998;
            justify-content: center;
            align-items: center;
        }
        .chart-modal-overlay.active {
            display: flex;
        }
        .chart-modal-content {
            background: #fff;
            border-radius: 12px;
            width: 92vw;
            height: 90vh;
            max-width: 1400px;
            position: relative;
            overflow: hidden;
            padding: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        .chart-modal-close {
            position: absolute;
            top: 12px; right: 16px;
            font-size: 24px;
            color: #666;
            cursor: pointer;
            z-index: 10;
            line-height: 1;
        }
        .chart-modal-close:hover {
            color: #000;
        }
        .chart-modal-title {
            font-size: 16px;
            font-weight: 700;
            color: #333;
            margin-bottom: 12px;
        }

        /* Stat header tooltip icon */
        .stat-tip {
            cursor: help;
        }
        .stat-tip .tip-icon {
            display: inline-block;
            width: 13px; height: 13px;
            border-radius: 50%;
            background: rgba(205,167,53,0.3);
            color: #CDA735;
            font-size: 9px; font-weight: 700;
            line-height: 13px; text-align: center;
            margin-left: 3px; vertical-align: middle;
        }
        /* Floating tooltip (appended to body) */
        .stat-tip-popup {
            position: fixed;
            background: #2d2d2d; color: #f0f0f0;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 11px; font-weight: 400;
            line-height: 1.5; width: 220px;
            text-align: left; z-index: 99999;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            white-space: normal;
            pointer-events: none;
        }
        .stat-tip-popup::before {
            content: '';
            position: absolute; top: -6px;
            left: 50%; transform: translateX(-50%);
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-bottom: 6px solid #2d2d2d;
        }
        .stat-tip-popup .tip-title {
            font-weight: 700; font-size: 12px;
            color: #CDA735; margin-bottom: 4px;
        }
        .stat-tip-popup .tip-scale {
            margin-top: 6px; font-size: 10px; color: #aaa;
        }

        .usage-table-wrap {
            width: 100%;
            max-width: 100%;
            overflow-x: auto !important;
            overflow-y: hidden !important;
            display: block;
            -webkit-overflow-scrolling: touch;
            padding-bottom: 6px;
        }

        .usage-table-wrap > div,
        .usage-table-wrap .table-responsive,
        .usage-table-wrap .dataframe_container {
            display: block;
            width: 100%;
        }

        .usage-table-wrap table,
        .usage-table-wrap .dataframe {
            width: 100% !important;
            min-width: 100% !important;
            border-collapse: collapse !important;
            table-layout: auto !important;
            white-space: nowrap;
        }

        /* Header */
        .usage-table-wrap thead th {
            background: #111 !important;
            color: #fff !important;
            font-weight: 800 !important;
            border: 1px solid #444 !important;
        }

        /* Body cells */
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

        /* Cell formatting */
        .usage-table-wrap th,
        .usage-table-wrap td {
            padding: 4px 6px;
            font-size: 12px;
            text-align: right;
        }

        /* First column */
        .usage-table-wrap th:first-child,
        .usage-table-wrap td:first-child {
            text-align: left;
            min-width: 140px;
            padding-right: 6px;
        }

        /* All other columns */
        .usage-table-wrap th:not(:first-child),
        .usage-table-wrap td:not(:first-child) {
            text-align: right;
            min-width: 88px;
        }

        /* Slightly wider metrics */
        .usage-table-wrap th:nth-child(7),
        .usage-table-wrap td:nth-child(7),
        .usage-table-wrap th:nth-child(8),
        .usage-table-wrap td:nth-child(8),
        .usage-table-wrap th:nth-child(9),
        .usage-table-wrap td:nth-child(9),
        .usage-table-wrap th:nth-child(10),
        .usage-table-wrap td:nth-child(10) {
            min-width: 95px;
        }

        /* Prediction tab table: slightly larger readability */
        .prediction-table-wrap th,
        .prediction-table-wrap td {
            font-size: 13px;
        }

        .table-title {
            font-size: 16px;
            font-weight: 900;
            margin: 6px 0 10px 0;
            text-align: center;
        }

        /* ---------------- Comparison tab ---------------- */
        .cmp-filter-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin-bottom: 14px;
            align-items: end;
            max-width: 1500px;
            margin-left: auto;
            margin-right: auto;
        }

        .cmp-filter-row .shiny-input-container {
            margin-bottom: 0 !important;
        }

        .cmp-filter-row .shiny-input-container > label {
            margin-bottom: 6px;
            font-weight: 800;
        }

        .cmp-vs-title {
            font-size: 26px;
            font-weight: 900;
            text-align: center;
            margin: 6px 0 16px 0;
            color: #222222;
        }

        .cmp-summary-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 14px;
        }

        .cmp-card {
            background: #ffffff;
            border: 1px solid #d6d6d6;
            border-radius: 12px;
            padding: 16px 18px;
        }

        .cmp-card-header {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 12px;
        }

        .cmp-card-logo {
            width: 54px;
            height: 54px;
            min-width: 54px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            background: #f7f7f7;
            border: 1px solid #ececec;
            font-size: 24px;
            font-weight: 900;
            color: #DDB945;
        }

        .cmp-card-name {
            font-size: 18px;
            font-weight: 900;
            margin: 0;
            color: #222222;
        }

        .cmp-card-subtitle {
            font-size: 13px;
            color: #666666;
            margin-top: 2px;
        }

        .cmp-chip-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 10px 0 14px 0;
        }

        .cmp-chip {
            background: #f3f3f3;
            border: 1px solid #e0e0e0;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 700;
            color: #333333;
        }

        .cmp-stat-strip {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 0;
            background: #f7f7f7;
            border: 1px solid #e1e1e1;
            border-radius: 12px;
        }

        .cmp-stat-box {
            padding: 12px 10px;
            text-align: center;
            border-right: 1px solid #e1e1e1;
        }

        .cmp-stat-box:last-child {
            border-right: none;
        }

        .cmp-stat-label {
            font-size: 13px;
            color: #666666;
            margin-bottom: 6px;
            font-weight: 700;
        }

        .cmp-stat-value {
            font-size: 18px;
            font-weight: 900;
            color: #222222;
        }

        .cmp-table-card {
            background: #ffffff;
            border: 1px solid #d6d6d6;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 14px;
        }

        .cmp-grid-bottom {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        .cmp-plot-card {
            background: #ffffff;
            border: 1px solid #d6d6d6;
            border-radius: 12px;
            padding: 10px;
        }

        .cmp-section-title {
            font-size: 16px;
            font-weight: 900;
            color: #222222;
            margin: 0 0 10px 4px;
            letter-spacing: 0.2px;
        }

        .cmp-table-wrap table,
        .cmp-table-wrap thead,
        .cmp-table-wrap tbody,
        .cmp-table-wrap tr,
        .cmp-table-wrap th,
        .cmp-table-wrap td,
        .cmp-table-wrap table.table,
        .cmp-table-wrap .table,
        .cmp-table-wrap .table > :not(caption) > * > * {
            background-color: transparent !important;
        }

        .cmp-table-wrap table {
            width: 100% !important;
            border-collapse: collapse !important;
            table-layout: fixed !important;
        }

        .cmp-table-wrap table.table thead th,
        .cmp-table-wrap .table thead th {
            background: #111 !important;
            color: #fff !important;
            font-weight: 800 !important;
            border: 1px solid #444 !important;
            border-bottom: 2px solid #444 !important;
            text-align: center !important;
            white-space: normal !important;
            line-height: 1.2 !important;
        }

        .cmp-table-wrap tbody td {
            border: 1px solid #cfcfcf !important;
            color: #111 !important;
            font-size: 13px !important;
            padding: 8px 8px !important;
        }

        .cmp-table-wrap tbody tr:nth-child(odd) td {
            background: #f7f7f7 !important;
        }

        .cmp-table-wrap tbody tr:nth-child(even) td {
            background: #ffffff !important;
        }

        .cmp-table-wrap th:nth-child(1),
        .cmp-table-wrap td:nth-child(1) {
            text-align: left !important;
            width: 34%;
        }

        .cmp-table-wrap th:nth-child(2),
        .cmp-table-wrap td:nth-child(2),
        .cmp-table-wrap th:nth-child(3),
        .cmp-table-wrap td:nth-child(3) {
            text-align: center !important;
            width: 33%;
        }

        /* ---------------- Team summary table ---------------- */
        .team-summary-wrap{
            margin-top: 14px;
            margin-bottom: 18px;
            background: #ffffff;
            border: 1px solid #d9d9d9;
            border-radius: 10px;
            overflow: auto;
            height: auto;
        }

        .team-summary-title{
            font-size: 20px;
            font-weight: 700;
            text-align: center;
            padding: 10px 12px;
            background: #f7f7f7;
            border-bottom: 1px solid #d9d9d9;
            color: #111111;
        }

        .team-summary-table{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }

        .team-summary-table th,
        .team-summary-table td{
            border: 1px solid #d9d9d9;
            padding: 10px 12px;
            text-align: center;
            font-size: 16px;
        }

        .team-summary-table th{
            font-weight: 700;
            background: #f3f3f3;
            white-space: nowrap;
        }

        .team-summary-table th:first-child,
        .team-summary-table td:first-child{
            text-align: left;
            font-weight: 700;
            width: 140px;
        }

        .team-summary-table th:nth-child(2){width: 140px;}
        .team-summary-table th:nth-child(3){width: 140px;}
        .team-summary-table th:nth-child(4){width: 140px;}
        .team-summary-table th:nth-child(5){width: 140px;}

        .team-summary-purdue{
            background: #f8f5e6 !important;
        }

        .team-summary-opponent{
            background: #f8f8f8 !important;
        }

        /* ---- Batter profile ---- */
        .bat-line-wrap {
            background: #ffffff;
            border: 1px solid #d6d6d6;
            border-radius: 10px;
            overflow-x: auto;
            margin-bottom: 14px;
        }
        .bat-line-wrap table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            min-width: 600px;
        }
        .bat-line-wrap thead th {
            background: #111111 !important;
            color: #ffffff !important;
            font-size: 13px !important;
            font-weight: 800 !important;
            text-align: center !important;
            padding: 8px 4px !important;
            border: 1px solid #333 !important;
        }
        .bat-line-wrap tbody td {
            font-size: 14px;
            font-weight: 600;
            text-align: center;
            padding: 10px 4px;
            border: 1px solid #e0e0e0;
            color: #111111;
        }
        .bat-line-wrap tbody td.bat-hl   { color: #185FA5; }
        .bat-line-wrap tbody td.bat-good { color: #0F6E56; }
        .bat-line-wrap tbody td.bat-warn { color: #854F0B; }
        .bat-line-wrap tbody tr:nth-child(odd) td  { background: #f7f7f7; }
        .bat-line-wrap tbody tr:nth-child(even) td { background: #ffffff; }

        
    """),

    # Purdue header (logo left, title centered)
    ui.tags.div(
        ui.tags.div(
            ui.tags.div(
                ui.tags.img(src=PURDUE_LOGO_SRC, alt="Purdue Logo"),
                class_="header-left",
            ),
            ui.tags.div("Softball Dashboard", class_="header-title"),
            ui.tags.div(class_="header-right"),
            class_="header-inner",
        ),
        class_="top-header",
    ),

    ui.tags.div(
        ui.tags.div(
            ui.tags.h4("Filters"),
            ui.tags.div(
                ui.tags.div("Season", class_="filter-title", style="margin-bottom: 6px;"),
                ui.input_select(
                    "season_choice",
                    "",
                    choices={
                        "spring_2026": "Spring 2026",
                        "all": "All Data",
                        "custom": "Custom Date Range",
                    },
                    
                    selected=INITIAL_SEASON_SELECTION,
                ),
            ),

            ui.tags.div(
                ui.tags.div("Date Range", class_="filter-title", style="margin-bottom: 6px;"),
                ui.input_date("date_start", "", value=_date_start_value),
                ui.input_date("date_end", "", value=_date_end_value),
                class_="date-range-row",
            ),

            ui.input_select(
                "data_source",
                "Data Source",
                choices={
                    "trackman":   "Trackman",
                    "hittrax":    "HitTrax",
                    "rapsodo":    "Rapsodo",
                    "collective": "Collective",
                },
                selected="trackman",
            ),

            ui.input_selectize(
                "team",
                "Team Name",
                choices={},
                selected=PURDUE_CODE,
                options={
                    "placeholder": "Search team name...",
                    "maxOptions": 300,
                    "openOnFocus": True,
                },
            ),
            ui.input_select(
                "session_type",
                "Session Type",
                choices={
                    "all": "All",
                    "bullpen": "Bullpen",
                    "batting_practice": "Batting Practice",
                    "scrimmage": "Scrimmage",
                    "live": "Live Game",
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
            ui.input_select(
                "batter_side",
                "Batter Side",
                choices={
                    "all": "Combined View",
                    "right": "vs Right Handed",
                    "left": "vs Left Handed",
                },
                selected="all",
            ),

            class_="sidebar",
        ),

        ui.tags.div(
            ui.output_ui("main_tabs"),
            class_="main-area",
        ),

        # Chart expand modal
        ui.tags.div(
            ui.tags.div(
                ui.tags.span("×", class_="chart-modal-close", onclick="closeChartModal()"),
                ui.tags.div("", class_="chart-modal-title", id="chart-modal-title"),
                ui.tags.div(id="chart-modal-body", style="width:100%;height:calc(100% - 50px);overflow:hidden;"),
                class_="chart-modal-content",
            ),
            class_="chart-modal-overlay",
            id="chart-modal-overlay",
            onclick="if(event.target===this)closeChartModal()",
        ),

        ui.tags.script(ui.HTML("""
        function expandChart(title, sourceId) {
            var src = document.getElementById(sourceId);
            if (!src) return;
            var overlay = document.getElementById('chart-modal-overlay');
            var body = document.getElementById('chart-modal-body');
            var titleEl = document.getElementById('chart-modal-title');
            titleEl.textContent = title;
            body.innerHTML = '';

            // Check for plotly chart
            var plotlyDiv = src.querySelector('.js-plotly-plot');
            if (plotlyDiv && window.Plotly) {
                // Show modal first so dimensions are available
                overlay.classList.add('active');
                var newDiv = document.createElement('div');
                newDiv.style.width = '100%';
                newDiv.style.height = '100%';
                body.appendChild(newDiv);
                var data = JSON.parse(JSON.stringify(plotlyDiv.data));
                var layout = JSON.parse(JSON.stringify(plotlyDiv.layout));
                // Use actual rendered dimensions after modal is visible
                setTimeout(function() {
                    layout.width = body.offsetWidth;
                    layout.height = body.offsetHeight;
                    layout.margin = {l:60, r:30, t:10, b:60};
                    layout.autosize = false;
                    Plotly.newPlot(newDiv, data, layout, {
                        displayModeBar: true,
                        displaylogo: false,
                        scrollZoom: true,
                        modeBarButtonsToRemove: ['toImage','select2d','lasso2d',
                            'hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines']
                    });
                }, 50);
                return;  // already showed overlay above
            } else {
                // Matplotlib — clone image and scale to fill modal
                overlay.classList.add('active');
                setTimeout(function() {
                    body.innerHTML = '';
                    body.style.display = 'flex';
                    body.style.flexDirection = 'column';
                    body.style.justifyContent = 'center';
                    body.style.alignItems = 'center';
                    body.style.overflow = 'hidden';
                    body.style.background = '#f7f7f7';

                    // If expanding pie chart, add the pitch legend at top
                    if (sourceId === 'pie') {
                        var legend = document.getElementById('movement_legend');
                        if (legend) {
                            var legendClone = legend.cloneNode(true);
                            legendClone.style.marginBottom = '10px';
                            legendClone.style.flexShrink = '0';
                            body.appendChild(legendClone);
                        }
                    }

                    var img = src.querySelector('img');
                    if (img) {
                        var newImg = img.cloneNode(true);
                        newImg.style.height = 'calc(100% - 50px)';
                        newImg.style.width = 'auto';
                        newImg.style.maxWidth = '100%';
                        newImg.style.objectFit = 'contain';
                        newImg.style.flexShrink = '1';
                        body.appendChild(newImg);
                    } else {
                        body.innerHTML = src.innerHTML;
                    }
                }, 50);
                return;
            }
        }
        function closeChartModal() {
            var body = document.getElementById('chart-modal-body');
            // Clean up plotly to avoid memory leaks
            var plots = body.querySelectorAll('.js-plotly-plot');
            plots.forEach(function(p) { if(window.Plotly) Plotly.purge(p); });
            document.getElementById('chart-modal-overlay').classList.remove('active');
            body.innerHTML = '';
        }
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeChartModal();
        });
        """)),

        ui.tags.script(ui.HTML("""
        var STAT_TIPS = {
            'BA': {title:'Batting Average', desc:'Hits divided by at-bats. The classic measure of hitting ability.'},
            'OBP': {title:'On-Base Percentage', desc:'How often the batter reaches base safely. Includes hits, walks, and HBP.'},
            'SLG': {title:'Slugging Percentage', desc:'Total bases divided by at-bats. Measures raw power output.'},
            'OPS': {title:'On-Base Plus Slugging', desc:'OBP + SLG combined. Quick overall offensive value measure.'},
            'wOBA': {title:'Weighted On-Base Average', desc:'Values all ways of reaching base by their actual run value. Most accurate single hitting stat.'},
            '__passive__': {title:'Passive Hitters', desc:'Batters whose overall swing rate is below the median of all batters in the current dataset (min 10 pitches seen).'},
            '__aggressive__': {title:'Aggressive Hitters', desc:'Batters whose overall swing rate is above the median of all batters in the current dataset (min 10 pitches seen).'},
            '__out_rate__': {title:'Out Rate by Zone', desc:'Outs (balls in play + strikeouts) divided by plate appearances ending with a pitch in that zone. Shows where the batter is most vulnerable.'}
        };
        var _tipPopup = null;
        function _showTip(e) {
            var th = e.currentTarget;
            var key = th.getAttribute('data-stat-key');
            if (!key || !STAT_TIPS[key]) return;
            var tip = STAT_TIPS[key];
            if (_tipPopup) _tipPopup.remove();
            _tipPopup = document.createElement('div');
            _tipPopup.className = 'stat-tip-popup';
            _tipPopup.innerHTML =
                '<div class="tip-title">' + tip.title + '</div>' +
                tip.desc;
            document.body.appendChild(_tipPopup);
            var rect = th.getBoundingClientRect();
            _tipPopup.style.left = (rect.left + rect.width/2 - 110) + 'px';
            _tipPopup.style.top = (rect.bottom + 8) + 'px';
        }
        function _hideTip() {
            if (_tipPopup) { _tipPopup.remove(); _tipPopup = null; }
        }
        function addStatTooltips() {
            document.querySelectorAll('.usage-table-wrap th').forEach(function(th) {
                var txt = th.textContent.trim();
                if (STAT_TIPS[txt] && !th.classList.contains('stat-tip')) {
                    th.classList.add('stat-tip');
                    th.setAttribute('data-stat-key', txt);
                    th.innerHTML = txt + '<span class="tip-icon">i</span>';
                    th.addEventListener('mouseenter', _showTip);
                    th.addEventListener('mouseleave', _hideTip);
                }
            });
            // Also attach to pre-marked stat-tip elements (e.g. approach table headers)
            document.querySelectorAll('.stat-tip[data-stat-key]').forEach(function(el) {
                if (!el._tipBound) {
                    el._tipBound = true;
                    el.addEventListener('mouseenter', _showTip);
                    el.addEventListener('mouseleave', _hideTip);
                }
            });
        }
        $(document).on('shiny:value', function() { setTimeout(addStatTooltips, 200); });
        $(document).ready(function() { setTimeout(addStatTooltips, 500); });
        """)),

        class_="layout-main",
    )
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):
    selected_pitch = reactive.Value("")
    selected_hittrax_session = reactive.Value("")  # ISO date string of selected session, or ""
    updating_dates_from_season = reactive.Value(False)

    @reactive.calc
    def date_range_invalid():
        start, end = input.date_start(), input.date_end()
        return start is not None and end is not None and start > end

    @reactive.calc
    def selected_range_has_data():
        start = input.date_start()
        end = input.date_end()

        if start is None or end is None or not csv_paths_with_dates:
            return False
        if start > end:
            return False

        for _, _, dmin, dmax in csv_paths_with_dates:
            if dmin <= end and dmax >= start:
                return True
        return False

    @reactive.effect
    def _update_dates_from_season():
        season = input.season_choice()

        if global_date_min is None or global_date_max is None:
            return

        if season == "custom":
            return

        if season == "all":
            start, end = global_date_min, global_date_max
        else:
            season_start, season_end = SEASON_DATE_MAP.get(
                season, (global_date_min, global_date_max)
            )
            start, end = clamp_date_range(
                season_start, season_end, global_date_min, global_date_max
            )

        updating_dates_from_season.set(True)

        if start is None or end is None:
            ui.notification_show(
                f"No data found for {season.replace('_', ' ').title()}.",
                type="warning",
                duration=3,
            )
        else:
            ui.update_date("date_start", value=start, session=session)
            ui.update_date("date_end", value=end, session=session)

        updating_dates_from_season.set(False)

    @reactive.effect
    def _update_date_bounds():
        start = input.date_start()
        end = input.date_end()

        if updating_dates_from_season.get():
            return

        if global_date_min is None or global_date_max is None:
            return
        if start is None or end is None:
            return
        if start > end:
            return

        ui.update_date(
            "date_start",
            min=global_date_min,
            max=min(end, global_date_max),
            session=session,
        )
        ui.update_date(
            "date_end",
            min=max(start, global_date_min),
            max=global_date_max,
            session=session,
        )

    @reactive.effect
    def _switch_season_to_custom_on_manual_date_edit():
        start = input.date_start()
        end = input.date_end()
        season = input.season_choice()

        if start is None or end is None:
            return

        if updating_dates_from_season.get():
            return

        if season == "custom":
            return

        if season == "all":
            expected_start, expected_end = global_date_min, global_date_max
        else:
            season_start, season_end = SEASON_DATE_MAP.get(
                season, (global_date_min, global_date_max)
            )
            expected_start, expected_end = clamp_date_range(
                season_start, season_end, global_date_min, global_date_max
            )

        if expected_start is None or expected_end is None:
            return

        if start != expected_start or end != expected_end:
            ui.update_select("season_choice", selected="custom", session=session)


    @reactive.effect
    def _notify_no_data_for_selected_range():
        season = input.season_choice()
        start = input.date_start()
        end = input.date_end()

        if start is None or end is None:
            return
        if date_range_invalid():
            return
        if selected_range_has_data():
            return

        msg = (
            f"No data found for {season.replace('_', ' ').title()}."
            if season not in ("all", "custom")
            else "No data found for the selected date range."
        )

        ui.notification_show(
            msg,
            type="warning",
            duration=3,
        )

    @reactive.calc
    def current_df():
        start = input.date_start()
        end = input.date_end()

        if start is None or end is None:
            return None
        if start > end:
            return None
        if MASTER_DF.empty:
            return None

        # Use pre-filtered source slice so we only pay the date-range filter cost
        src = input.data_source()
        base = SOURCE_SLICES.get(src, MASTER_DF) if src in SOURCE_SLICES else MASTER_DF

        if base is None or base.empty:
            return base

        df = base[
            (base["DateOnly"] >= start) &
            (base["DateOnly"] <= end)
        ]

        return df

    @reactive.effect
    def _warn_unsupported_source():
        src = input.data_source()
        if src not in ("trackman", "rapsodo", "collective", "hittrax"):
            ui.notification_show(
                f"'{input.data_source()}' data is not yet available.",
                type="warning",
                duration=4,
            )
            ui.update_select("data_source", selected="trackman", session=session)
        elif src == "rapsodo":
            # Rapsodo is bullpen only, no batter data
            ui.update_select(
                "session_type",
                choices={"bullpen": "Bullpen"},
                selected="bullpen",
                session=session,
            )
            ui.update_radio_buttons("player_type", selected="pitcher", session=session)
        elif src == "hittrax":
            # HitTrax is batting practice only, batter profile only
            ui.update_select(
                "session_type",
                choices={"batting_practice": "Batting Practice"},
                selected="batting_practice",
                session=session,
            )
        else:
            # Trackman or Collective — restore full session type choices
            ui.update_select(
                "session_type",
                choices={
                    "all": "All",
                    "bullpen": "Bullpen",
                    "batting_practice": "Batting Practice",
                    "scrimmage": "Scrimmage",
                    "live": "Live Game",
                },
                session=session,
            )

    @reactive.effect
    def _update_team_choices():
        src = input.data_source()
        if src == "hittrax":
            # HitTrax is Purdue-only
            ui.update_selectize(
                "team",
                choices={PURDUE_CODE: TEAM_NAME_MAP.get(PURDUE_CODE, PURDUE_CODE)},
                selected=PURDUE_CODE,
                session=session,
            )
            return

        df = current_df()
        if df is None or df.empty:
            ui.update_selectize("team", choices={}, session=session)
            return

        teams = set()
        for col in ["PitcherTeam", "BatterTeam"]:
            if col in df.columns:
                # .unique() is much faster than .tolist() on large columns
                vals = df[col].dropna().astype(str).str.strip().unique()
                teams.update(v for v in vals if v)

        teams = sorted(teams)
        choices = {t: TEAM_NAME_MAP.get(t, t) for t in teams}
        choices = dict(sorted(choices.items(), key=lambda x: x[1]))
        default_team = PURDUE_CODE if PURDUE_CODE in teams else (teams[0] if teams else None)
        ui.update_selectize(
            "team",
            choices=choices,
            selected=default_team,
            session=session,
        )

    @reactive.effect
    def _force_session_all_for_non_purdue():
        team = input.team()
        if team and not is_purdue_team(team):
            ui.update_select("session_type", selected="all", session=session)

    @reactive.effect
    def _update_player_choices():
        src = input.data_source()
        if src == "hittrax":
            # HitTrax: populate from HITTRAX_DF by player name (used as id)
            if HITTRAX_DF.empty:
                ui.update_select("player", choices={"": "—"}, label="Player Name", session=session)
                return
            start = input.date_start()
            end = input.date_end()
            hdf = HITTRAX_DF
            if start is not None and end is not None:
                hdf = hdf[(hdf["DateOnly"] >= start) & (hdf["DateOnly"] <= end)]
            names = sorted(hdf["Player"].dropna().unique().tolist())
            choices = {n: format_display_name(n) or n for n in names}
            ui.update_select(
                "player",
                choices=choices if choices else {"": "—"},
                label="Player Name",
                selected=list(choices.keys())[0] if choices else None,
                session=session,
            )
            return

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
            if "BatterTeam" in df.columns:
                df = df[df["BatterTeam"].astype(str).str.strip() == str(team)]
            if "BatterId" not in df.columns or "Batter" not in df.columns:
                ui.update_select("player", choices={"": "—"}, label="Player Name", session=session)
                return
            lookup = (
                df[["BatterId", "Batter"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["Batter", "BatterId"])
            )
            choices = {
                str(r.BatterId): (format_display_name(r.Batter) or str(r.BatterId))
                for r in lookup.itertuples(index=False)
            }
            ui.update_select(
                "player",
                choices=choices,
                label="Player Name",
                selected=list(choices.keys())[0] if choices else None,
                session=session,
            )
            
    @reactive.effect
    def _update_handedness_label():
        ptype = input.player_type()
        if ptype == "batter":
            ui.update_select(
                "batter_side",
                label="Pitcher Hand",
                choices={
                    "all": "Combined View",
                    "right": "vs Right Handed",
                    "left": "vs Left Handed",
                },
                session=session,
            )
        else:
            ui.update_select(
                "batter_side",
                label="Batter Side",
                choices={
                    "all": "Combined View",
                    "right": "vs Right Handed",
                    "left": "vs Left Handed",
                },
                session=session,
            )

    @reactive.effect
    @reactive.event(input.reset_pitch)
    def _clear_clicked_pitch():
        selected_pitch.set("")

    @reactive.effect
    @reactive.event(input.clicked_pitch)
    def _sync_legend_click():
        val = input.clicked_pitch()
        if not val:
            return

        if val == "__reset__" or val == selected_pitch.get():
            selected_pitch.set("")
        else:
            selected_pitch.set(str(val))



    @reactive.calc
    def batter_data():
        df = current_df()
        team = input.team()
        bid = input.player() if input.player_type() == "batter" else None
        if df is None or not team or not bid:
            return None
        if "BatterTeam" in df.columns:
            df = df[df["BatterTeam"].astype(str).str.strip() == str(team)]
        df = apply_session_filter_for_team(df, team, input.session_type())
        df = df[df["BatterId"].astype(str) == str(bid)]

        # Pitcher handedness filter (reuses batter_side input)
        hand = input.batter_side()
        if hand != "all" and "PitcherThrows" in df.columns:
            side_series = df["PitcherThrows"].astype(str).str.strip().str.lower()
            if hand == "right":
                df = df[side_series.isin(["right", "r"])]
            elif hand == "left":
                df = df[side_series.isin(["left", "l"])]

        return df

    @reactive.calc
    def batter_summary_text():
        data = batter_data()
        if data is None or data.empty:
            return None
        name = format_display_name(
            data["Batter"].iloc[0] if "Batter" in data.columns else ""
        ) or "Batter"
        side = str(data["BatterSide"].iloc[0]).strip() \
               if "BatterSide" in data.columns else ""
        side = "" if side.lower() in ("nan", "") else side
        pa   = compute_batter_stats(data, input.player())["PA"]

        hand_label = {
            "all": "Combined View",
            "right": "vs RHP",
            "left": "vs LHP",
        }.get(input.batter_side(), "Combined View")

        parts = [name]
        if side:
            parts.append(f"Bats: {side}")
        parts.append(hand_label)
        parts.append(f"{pa} PA")
        return " | ".join(parts)

    @reactive.calc
    def batter_pitch_colors():
        data = batter_data()
        if data is None or data.empty or PITCH_TYPE_COL not in data.columns:
            return {}
        return build_pitch_color_map(data[PITCH_TYPE_COL].dropna().unique())

    @reactive.calc
    def batter_pitch_order():
        data = batter_data()
        if data is None or data.empty or PITCH_TYPE_COL not in data.columns:
            return []
        return (
            data.loc[is_valid_pitch_type(data[PITCH_TYPE_COL]), PITCH_TYPE_COL]
            .astype(str).value_counts().index.tolist()
        )

    @reactive.calc
    def batter_selected_pitch():
        """Return selected_pitch only if it's valid for current batter's data."""
        sel = selected_pitch.get()
        if not sel:
            return ""
        order = batter_pitch_order()
        return sel if sel in order else ""

    @reactive.calc
    def batter_percentile_thresholds():
        """Compute percentile thresholds for BA/OBP/SLG/OPS/wOBA across all batters.

        Vectorized with groupby for speed — previously a Python for-loop over
        every BatterId, which dominated load time when changing Data Source.
        """
        df = current_df()
        if df is None or df.empty or "BatterId" not in df.columns:
            return {}

        MIN_PA = 10
        TERMINAL = {"Single","Double","Triple","HomeRun","Out","FieldersChoice",
                    "Error","Walk","Strikeout","HitByPitch","SacrificeFly",
                    "SacrificeBunt","Sacrifice","CatcherInterference"}

        # Work on a minimal projection
        cols_needed = ["BatterId", "PlayResult", "PitchCall", "PitchofPA"]
        cols_present = [c for c in cols_needed if c in df.columns]
        d = df[cols_present].copy()
        d["BatterId"] = d["BatterId"].astype(str)

        pr = d["PlayResult"].astype(str).str.strip() if "PlayResult" in d.columns else pd.Series("", index=d.index)
        pc = d["PitchCall"].astype(str).str.strip()  if "PitchCall"  in d.columns else pd.Series("", index=d.index)
        d["is_terminal"] = pr.isin(TERMINAL) | pc.eq("HitByPitch")

        # PA count per batter — prefer PitchofPA == 1 if available
        if "PitchofPA" in d.columns:
            pop = pd.to_numeric(d["PitchofPA"], errors="coerce")
            d["is_pa"] = (pop == 1)
            pa_counts = d.groupby("BatterId")["is_pa"].sum().astype(int)
        else:
            pa_counts = d.groupby("BatterId")["is_terminal"].sum().astype(int)

        eligible = pa_counts[pa_counts >= MIN_PA].index
        if len(eligible) < 5:
            return {}

        # Filter to eligible batters + terminal rows only (for hit counting)
        td = d[d["BatterId"].isin(eligible) & d["is_terminal"]].copy()
        td_pr = td["PlayResult"].astype(str).str.strip() if "PlayResult" in td.columns else pd.Series("", index=td.index)
        td_pc = td["PitchCall"].astype(str).str.strip()  if "PitchCall"  in td.columns else pd.Series("", index=td.index)

        td["Single"]  = td_pr.eq("Single")
        td["Double"]  = td_pr.eq("Double")
        td["Triple"]  = td_pr.eq("Triple")
        td["HR"]      = td_pr.eq("HomeRun")
        td["BB"]      = td_pr.eq("Walk")
        td["K"]       = td_pr.eq("Strikeout")
        td["HBP"]     = td_pr.eq("HitByPitch") | td_pc.eq("HitByPitch")

        agg = td.groupby("BatterId")[
            ["Single", "Double", "Triple", "HR", "BB", "K", "HBP"]
        ].sum().astype(int)

        # Align PA
        agg["PA"] = pa_counts.reindex(agg.index).fillna(0).astype(int)
        agg["H"]  = agg["Single"] + agg["Double"] + agg["Triple"] + agg["HR"]
        agg["AB"] = (agg["PA"] - agg["BB"] - agg["HBP"]).clip(lower=0)
        agg["TB"] = agg["Single"] + 2*agg["Double"] + 3*agg["Triple"] + 4*agg["HR"]

        # Drop batters with 0 AB (we want rate stats)
        agg = agg[agg["AB"] > 0]
        if len(agg) < 5:
            return {}

        agg["BA"]  = agg["H"] / agg["AB"]
        agg["OBP"] = (agg["H"] + agg["BB"] + agg["HBP"]) / agg["PA"]
        agg["SLG"] = agg["TB"] / agg["AB"]
        agg["OPS"] = agg["OBP"] + agg["SLG"]
        agg["wOBA"] = (
            0.690*agg["BB"] + 0.720*agg["HBP"] + 0.880*agg["Single"]
            + 1.242*agg["Double"] + 1.569*agg["Triple"] + 2.007*agg["HR"]
        ) / agg["PA"]

        thresholds = {}
        for col in ["BA", "OBP", "SLG", "OPS", "wOBA"]:
            vals = agg[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) < 5:
                continue
            thresholds[col] = {
                "p33": float(vals.quantile(0.33)),
                "p67": float(vals.quantile(0.67)),
            }
        return thresholds

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
        df = df[df["PitcherId"].astype(str) == str(pid)]

        # Batter handedness filter
        batter_side = input.batter_side()

        if batter_side != "all" and "BatterSide" in df.columns:
            side_series = df["BatterSide"].astype(str).str.strip().str.lower()

            if batter_side == "right":
                df = df[side_series.isin(["right", "r"])]
            elif batter_side == "left":
                df = df[side_series.isin(["left", "l"])]

        return df

    @reactive.calc
    def batter_swing_rate_median():
        """Median swing rate across all batters in current dataset (min 10 pitches)."""
        df = current_df()
        if df is None or df.empty or "BatterId" not in df.columns or "PitchCall" not in df.columns:
            return None
        SW = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        rates = []
        for bid, grp in df.groupby("BatterId"):
            n = len(grp)
            if n < 10:
                continue
            pc = grp["PitchCall"].astype(str).str.strip()
            sw = pc.isin(SW).sum()
            rates.append(sw / n)
        if not rates:
            return None
        return float(np.median(rates))

    @reactive.calc
    def pitch_colors():
        data = pitcher_data()
        if data is None or PITCH_TYPE_COL not in data.columns:
            return {}
        return build_pitch_color_map(data[PITCH_TYPE_COL].dropna().unique())

    @reactive.calc
    def pitch_order():
        data = pitcher_data()
        if data is None or data.empty or PITCH_TYPE_COL not in data.columns:
            return []
        return (
            data.loc[is_valid_pitch_type(data[PITCH_TYPE_COL]), PITCH_TYPE_COL]
            .astype(str).value_counts().index.tolist()
        )

    @reactive.calc
    def pitcher_loc_data():
        data = pitcher_data()
        if data is None or data.empty:
            return pd.DataFrame()
        d = data.copy()
        d["PlateLocSide"]   = pd.to_numeric(d["PlateLocSide"],   errors="coerce")
        d["PlateLocHeight"] = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
        return d[
            d["PlateLocSide"].notna()
            & d["PlateLocHeight"].notna()
            & is_valid_pitch_type(d[PITCH_TYPE_COL])
        ]


    @reactive.calc
    def pitcher_mov_data():
        data = pitcher_data()
        if data is None or data.empty:
            return pd.DataFrame()
        d = data.copy()
        d[X_MOV] = pd.to_numeric(d[X_MOV], errors="coerce")
        d[Y_MOV] = pd.to_numeric(d[Y_MOV], errors="coerce")
        return d[
            d[X_MOV].notna()
            & d[Y_MOV].notna()
            & is_valid_pitch_type(d[PITCH_TYPE_COL])
        ]

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

        side_label = {
            "all": "Combined View",
            "right": "vs RHB",
            "left": "vs LHB",
        }.get(input.batter_side(), "Combined View")

        if hand:
            return f"{name} | {hand} | {side_label} | {n_pitches} pitches"
        return f"{name} | {side_label} | {n_pitches} pitches"

    
    @reactive.calc
    def cached_pitch_metrics():
        data = pitcher_data()
        pid = input.player() if input.player_type() == "pitcher" else None
        return compute_pitch_metrics(data, pid)

    @reactive.calc
    def session_player_type_warning():
        session_val = input.session_type()
        ptype = input.player_type()
        src = input.data_source()
        if src == "hittrax" and ptype == "pitcher":
            return "⚠️  HitTrax only contains batter data. Switch Player Type to Batter to view the profile."
        if src == "rapsodo" and ptype == "batter":
            return "⚠️  Rapsodo only contains pitcher data. Switch Player Type to Pitcher to view the profile."
        if session_val == "bullpen" and ptype == "batter":
            return "⚠️  Bullpen sessions only contain pitcher data. Switch Player Type to Pitcher to view the profile."
        if session_val == "batting_practice" and ptype == "pitcher":
            return "⚠️  Batting Practice sessions only contain batter data. Switch Player Type to Batter to view the profile."
        return None


    # -----------------------------------------------------------------------
    # Prediction tab reactives
    # -----------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.retrain_prediction_models)
    def _retrain_prediction_models():
        prediction_pipeline.clear_prediction_cache(remove_disk=True)
        ui.notification_show(
            "Prediction models cache cleared. Models will retrain on next prediction refresh.",
            type="message",
            duration=4,
        )

    @reactive.calc
    def prediction_df():
        data = pitcher_data()
        if data is None or data.empty:
            return pd.DataFrame()
        return build_prediction_by_pitch_type(data)

    @reactive.calc
    def prediction_summary():
        return select_prediction_summary(prediction_df())

    @reactive.calc
    def ml_prediction_state():
        empty_st = {"use_ml": False, "message": "", "df": pd.DataFrame(),
                    "summary": {"best_strike": None, "best_putaway": None, "caution": None},
                    "training_note": "", "metrics_note": "", "warning_note": ""}
        if input.player_type() != "pitcher":
            return empty_st
        data = pitcher_data()
        if data is None or data.empty:
            return empty_st
        league = SOURCE_SLICES.get("trackman")
        if league is None or league.empty:
            if MASTER_DF is not None and not MASTER_DF.empty and "DataSource" in MASTER_DF.columns:
                league = MASTER_DF[MASTER_DF["DataSource"].astype(str) == "trackman"].reset_index(drop=True)
            else:
                league = MASTER_DF
        desc = build_prediction_by_pitch_type(data)
        return prediction_pipeline.compute_ml_prediction_bundle(
            pitcher_df=data,
            league_df=league,
            descriptive_by_ptype=desc,
            batter_side_filter=input.batter_side(),
            pitch_col=PITCH_TYPE_COL,
            zone_bounds=(ZONE_LEFT, ZONE_RIGHT, ZONE_BOTTOM, ZONE_TOP),
        )

    @reactive.calc
    def ml_prediction_df():
        return ml_prediction_state()["df"]

    @reactive.calc
    def ml_prediction_summary():
        return ml_prediction_state()["summary"]

    @output
    @render.ui
    def prediction_content():
        if input.player_type() != "pitcher":
            return ui.div("Prediction is currently available for pitcher view only.",
                          style="padding:20px;color:#666;text-align:center;font-size:16px;")
        if input.data_source() not in ("trackman",):
            return ui.div("⚠️  Prediction is available for Trackman data only. Switch Data Source to Trackman to view predictions.", style=(
                "padding:14px 18px; border-radius:8px; background:#fef9ec;"
                "border:1.5px solid #DDB945; color:#6b4e00;"
                "font-size:16px; font-weight:700; margin-top:12px;"
            ))

        pred = prediction_df()
        if pred is None or pred.empty:
            return ui.div("No prediction data available for the selected filters.",
                          style="padding:20px;color:#666;text-align:center;font-size:16px;")

        st = ml_prediction_state()
        use_ml = bool(st.get("use_ml"))
        summ = st["summary"] if use_ml else prediction_summary()
        fallback_note = (st.get("message") or "").strip()

        def _pct(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "—"
            return f"{float(x) * 100:.1f}%"

        def _short_explainer(key):
            if key == "best_strike":
                return "Consistently lands in the zone."
            if key == "best_putaway":
                return "Generates swings and misses late in counts."
            return "Gets hit hard more often when left over the plate."

        def _takeaway(key):
            if key == "best_strike":
                return "Most reliable pitch to get ahead in the count"
            if key == "best_putaway":
                return "Best swing-and-miss pitch in two-strike situations"
            return "Most likely to get hit hard if mislocated"

        def _when_to_use(key):
            if key == "best_strike":
                return ["Early counts (0-0, 1-0)", "When you need a strike"]
            if key == "best_putaway":
                return ["Two-strike counts", "After showing fastball"]
            return ["Use carefully in hitter's counts", "Avoid middle-middle location"]

        def _card_theme(key):
            if key == "best_strike":
                return ("#0f8a4a", "#f4fff8")
            if key == "best_putaway":
                return ("#b38b00", "#fffdf4")
            return ("#b91c1c", "#fff7f7")

        def _card(title, key):
            c = summ.get(key)
            border_color, bg_color = _card_theme(key)
            if not c or not c.get("pitch") or c.get("pitch") == "—":
                body = ui.div("Not enough data for this recommendation.", style="font-size:15px;color:#555;")
            elif c.get("is_ml"):
                warn = c.get("sample_warning") or ""
                warn_ui = ui.div(ui.tags.span("Sample note: ", style="font-weight:800;"), warn, style="font-size:13px;color:#854F0B;margin-top:8px;") if warn else None
                bullets = _when_to_use(key)
                body = ui.div(
                    ui.div(str(c["pitch"]), style="font-size:32px;font-weight:900;margin-bottom:6px;color:#111;"),
                    ui.div(_takeaway(key), style="font-size:15px;font-weight:900;color:#111;margin-bottom:8px;"),
                    ui.div(
                        f"Strike {_pct(c.get('pred_strike'))} | Whiff {_pct(c.get('pred_whiff'))} | Hard Contact Risk {_pct(c.get('pred_hard'))}",
                        style="font-size:15px;color:#222;margin-bottom:8px;",
                    ),
                    ui.div(_short_explainer(key), style="font-size:14px;color:#222;line-height:1.35;"),
                    ui.div("When to use:", style="font-size:13px;font-weight:900;color:#111;margin-top:10px;"),
                    ui.tags.ul(
                        ui.tags.li(bullets[0], style="font-size:13px;color:#222;"),
                        ui.tags.li(bullets[1], style="font-size:13px;color:#222;"),
                        style="margin:4px 0 0 18px;padding:0;",
                    ),
                    warn_ui,
                )
            else:
                body = ui.div("Historical summary only (ML unavailable).", style="font-size:14px;color:#666;")

            return ui.div(
                ui.div(title, style=f"font-size:14px;font-weight:900;letter-spacing:0.04em;color:{border_color};margin-bottom:8px;"),
                body,
                style=f"flex:1;min-width:220px;padding:16px;background:{bg_color};border:1px solid #d6d6d6;border-radius:10px;border-top:5px solid {border_color};",
            )

        strike_pitch = ((summ.get("best_strike") or {}).get("pitch") or "this pitch")
        putaway_pitch = ((summ.get("best_putaway") or {}).get("pitch") or "this pitch")
        caution_pitch = ((summ.get("caution") or {}).get("pitch") or "this pitch")
        strategy_ui = ui.div(
            ui.div("Pitching Strategy Summary:", style="font-size:15px;font-weight:900;color:#111;margin-bottom:4px;"),
            ui.div(
                f"This pitcher leans on the {str(strike_pitch).lower()} for control, "
                f"uses the {str(putaway_pitch).lower()} as the put-away pitch, "
                f"and should be cautious with the {str(caution_pitch).lower()} when behind in the count.",
                style="font-size:15px;color:#222;line-height:1.4;",
            ),
            style="padding:10px 12px;border:1px solid #d9d9d9;border-left:5px solid #111;border-radius:8px;background:#fcfcfc;margin-bottom:10px;",
        )

        train_note = (st.get("training_note") or "").strip()
        train_ui = ui.div(train_note, style="font-size:12px;color:#666;font-style:italic;margin-bottom:6px;") if (use_ml and train_note) else None
        met_note = (st.get("metrics_note") or "").strip()
        met_ui = ui.div(f"Validation: {met_note}", style="font-size:11px;color:#777;margin-bottom:6px;") if (use_ml and met_note) else None
        warn_note = (st.get("warning_note") or "").strip()
        warn_ui = ui.div(
            f"Estimate note: {warn_note}",
            style="font-size:12px;color:#854F0B;font-weight:700;margin-bottom:8px;",
        ) if (use_ml and warn_note) else None
        fb_ui = ui.div(fallback_note, style="font-size:14px;color:#854F0B;font-weight:700;margin-bottom:12px;") if (not use_ml and fallback_note) else None

        advanced_ui = ui.div()
        if use_ml:
            advanced_ui = ui.tags.details(
                ui.tags.summary("Advanced Metrics"),
                ui.div(
                    ui.tags.p(
                        "Model detail view: shows how each pitch profile shifts expected outcomes versus league-average baseline.",
                        style="font-size:13px;color:#666;margin:8px 0 4px 0;",
                    ),
                    ui.tags.p(
                        ui.tags.span("Green/right", style="color:#166534;font-weight:800;"),
                        " = increases the outcome | ",
                        ui.tags.span("Red/left", style="color:#991b1b;font-weight:800;"),
                        " = decreases the outcome | Values are directional effects, not raw probabilities.",
                        style="font-size:12px;color:#555;margin:0 0 8px 0;",
                    ),
                    ui.output_ui("prediction_advanced_impacts_ui"),
                    ui.output_ui("prediction_model_drivers_ui"),
                    ui.output_table("prediction_advanced_table"),
                    ui.tags.p(
                        "Use these values to compare relative strengths across this pitcher's options. Treat small gaps as equivalent tiers; prioritize cards and game context for final pitch calling.",
                        style="font-size:12px;color:#555;margin:8px 0 4px 0;",
                    ),
                ),
                style="margin-top:12px;padding:8px;border:1px solid #ddd;border-radius:8px;background:#fafafa;",
            )

        return ui.div(
            strategy_ui, train_ui, met_ui, warn_ui, fb_ui,
            ui.div(
                _card("Best strike / command pitch", "best_strike"),
                _card("Best put-away option", "best_putaway"),
                _card("Highest damage-risk pitch", "caution"),
                style="display:flex;flex-wrap:wrap;gap:14px;align-items:stretch;",
            ),
            advanced_ui,
            style="padding:4px 0 18px 0;",
            class_="prediction-tab-wrap",
        )

    @output
    @render.ui
    def prediction_table_section():
        if input.player_type() != "pitcher" or input.data_source() not in ("trackman",):
            return ui.div()
        return ui.div(
            ui.div("Summary by pitch type", style=(
                "font-size:17px;font-weight:900;margin:18px 0 10px 0;text-align:center;"
            )),
            ui.div(ui.output_table("prediction_table"), class_="usage-table-wrap prediction-table-wrap"),
        )

    @output
    @render.table
    def prediction_table():
        if input.player_type() != "pitcher" or input.data_source() not in ("trackman",):
            return pd.DataFrame()
        st = ml_prediction_state()
        if st.get("use_ml"):
            return prediction_pipeline.format_ml_prediction_table_display(st["df"], st["summary"])
        return format_prediction_table_display(prediction_df(), prediction_summary())

    @output
    @render.ui
    def prediction_advanced_impacts_ui():
        if input.player_type() != "pitcher" or input.data_source() not in ("trackman",):
            return ui.div()
        st = ml_prediction_state()
        if not st.get("use_ml") or st.get("df") is None or st.get("df").empty:
            return ui.div()

        adf = st["df"].copy()
        req_cols = [
            PITCH_TYPE_COL,
            "attack_score",
            "putaway_score",
            "danger_score",
            "composite_score",
            "stability_label",
        ]
        req_cols = [c for c in req_cols if c in adf.columns]
        if len(req_cols) < 5:
            return ui.div()

        adf = adf[req_cols].copy()
        adf = adf.sort_values("composite_score", ascending=False, kind="mergesort")

        max_abs = pd.to_numeric(
            adf[["attack_score", "putaway_score", "danger_score"]].stack(),
            errors="coerce",
        ).abs().max()
        if pd.isna(max_abs) or float(max_abs) <= 0:
            max_abs = 1.0

        def _fmt_signed(v):
            try:
                fv = float(v)
            except Exception:
                return "—"
            return f"{fv:+.3f}"

        low_stability_mask = adf.get("stability_label", pd.Series([""] * len(adf))).astype(str).str.lower().isin(
            ["low stability", "very low stability"]
        )
        has_low_stability = bool(low_stability_mask.any())

        def _chip_text(metric_col, metric_label, high_is_risk=False):
            vals = pd.to_numeric(adf[metric_col], errors="coerce")
            if vals.notna().sum() == 0:
                return f"No data for {metric_label.lower()}."
            idx = vals.idxmax()
            pitch = str(adf.loc[idx, PITCH_TYPE_COL])
            sval = _fmt_signed(adf.loc[idx, metric_col])
            if has_low_stability:
                if high_is_risk:
                    return f"{pitch}: leans toward higher {metric_label.lower()} ({sval})"
                return f"{pitch}: leans toward stronger {metric_label.lower()} ({sval})"
            if high_is_risk:
                return f"{pitch}: highest {metric_label.lower()} ({sval})"
            return f"{pitch}: strongest {metric_label.lower()} ({sval})"

        chip_style = (
            "display:inline-block;margin:4px 6px 4px 0;padding:6px 10px;"
            "border:1px solid #d5d5d5;border-radius:999px;background:#fff;font-size:12px;color:#222;"
        )
        chips = ui.div(
            ui.tags.span(_chip_text("attack_score", "control impact"), style=chip_style),
            ui.tags.span(_chip_text("putaway_score", "two-strike put-away impact"), style=chip_style),
            ui.tags.span(_chip_text("danger_score", "hard-contact risk impact", high_is_risk=True), style=chip_style),
            style="margin:4px 0 10px 0;",
        )
        low_sample_badge = ui.div()
        if has_low_stability:
            low_sample_badge = ui.div(
                "Low sample: directional only.",
                style=(
                    "display:inline-block;margin:0 0 10px 0;padding:5px 9px;"
                    "border-radius:999px;border:1px solid #f0b429;background:#fff7e6;"
                    "font-size:12px;font-weight:800;color:#8a5d00;"
                ),
            )

        def _impact_row(metric_label, value, positive_color, negative_color):
            try:
                fv = float(value)
            except Exception:
                fv = 0.0
            width_pct = max(0.0, min(50.0, 50.0 * abs(fv) / float(max_abs)))
            if fv >= 0:
                left_w, right_w = 0.0, width_pct
                left_bg, right_bg = "transparent", positive_color
            else:
                left_w, right_w = width_pct, 0.0
                left_bg, right_bg = negative_color, "transparent"
            return ui.div(
                ui.div(metric_label, style="min-width:210px;font-size:12px;color:#333;font-weight:700;"),
                ui.div(
                    ui.div(
                        ui.div(style=f"position:absolute;left:{50-left_w}%;top:0;height:100%;width:{left_w}%;background:{left_bg};"),
                        ui.div(style=f"position:absolute;left:50%;top:0;height:100%;width:{right_w}%;background:{right_bg};"),
                        ui.div(style="position:absolute;left:50%;top:0;height:100%;width:2px;background:#666;transform:translateX(-1px);"),
                        style="position:relative;height:12px;background:#f1f1f1;border:1px solid #d0d0d0;border-radius:999px;overflow:hidden;",
                    ),
                    style="flex:1;min-width:200px;",
                ),
                ui.div(_fmt_signed(fv), style="min-width:58px;text-align:right;font-size:12px;color:#222;font-family:monospace;"),
                style="display:flex;align-items:center;gap:10px;margin:6px 0;",
            )

        cards = []
        for _, row in adf.iterrows():
            pitch = str(row[PITCH_TYPE_COL])
            stab = str(row.get("stability_label", "")).strip().lower()
            is_low = stab in ("low stability", "very low stability")
            card_opacity = "0.72" if is_low else "1.0"
            stability_note = ui.div()
            if is_low:
                stability_note = ui.div(
                    "Low sample: directional only.",
                    style="font-size:11px;color:#8a5d00;font-weight:800;margin:3px 0 6px 0;",
                )
            cards.append(
                ui.div(
                    ui.div(pitch, style="font-size:13px;font-weight:900;color:#111;margin-bottom:4px;"),
                    stability_note,
                    _impact_row("Control impact", row["attack_score"], "#86c59a", "#e59a9a"),
                    _impact_row("Two-strike put-away impact", row["putaway_score"], "#86c59a", "#e59a9a"),
                    _impact_row("Hard-contact risk impact", row["danger_score"], "#e59a9a", "#86c59a"),
                    style=(
                        f"border:1px solid #e0e0e0;border-radius:8px;background:#fff;padding:8px 10px;"
                        f"margin-bottom:8px;opacity:{card_opacity};"
                    ),
                )
            )

        return ui.div(chips, low_sample_badge, *cards)

    @output
    @render.ui
    def prediction_model_drivers_ui():
        if input.player_type() != "pitcher" or input.data_source() not in ("trackman",):
            return ui.div()
        st = ml_prediction_state()
        if not st.get("use_ml"):
            return ui.div()
        summ = st.get("summary") or {}
        if not isinstance(summ, dict):
            return ui.div()

        def _drivers_card(title, card):
            if not isinstance(card, dict) or not card.get("pitch"):
                return None
            pos = [str(x) for x in card.get("drivers_pos", []) if str(x).strip()]
            neg = [str(x) for x in card.get("drivers_neg", []) if str(x).strip()]
            src = str(card.get("drivers_source", "Fallback profile drivers")).strip() or "Fallback profile drivers"
            stab = str(card.get("stability_label", "")).strip()
            if not pos:
                pos = ["limited positive signal"]
            if not neg:
                neg = ["limited counter-signal"]
            return ui.div(
                ui.div(
                    f"{title}: {card.get('pitch', '—')}",
                    style="font-size:13px;font-weight:900;color:#111;margin-bottom:4px;",
                ),
                ui.div(
                    ui.tags.span("Upward drivers: ", style="font-weight:800;color:#166534;"),
                    ", ".join(pos[:2]),
                    style="font-size:12px;color:#222;margin-bottom:2px;",
                ),
                ui.div(
                    ui.tags.span("Downward drivers: ", style="font-weight:800;color:#991b1b;"),
                    ", ".join(neg[:2]),
                    style="font-size:12px;color:#222;margin-bottom:4px;",
                ),
                ui.div(
                    f"Source: {src} | Stability: {stab or '—'}",
                    style="font-size:11px;color:#666;",
                ),
                style="border:1px solid #dfdfdf;background:#fff;border-radius:8px;padding:8px 10px;margin-bottom:8px;",
            )

        cards = [
            _drivers_card("Best command", summ.get("best_strike")),
            _drivers_card("Best put-away", summ.get("best_putaway")),
            _drivers_card("Highest risk", summ.get("caution")),
        ]
        cards = [c for c in cards if c is not None]
        if not cards:
            return ui.div()

        return ui.div(
            ui.div("Model Drivers (SHAP)", style="font-size:14px;font-weight:900;color:#111;margin:10px 0 6px 0;"),
            ui.div(
                "SHAP drivers explain why model probabilities moved; impact bars show decision-score ranking.",
                style="font-size:12px;color:#555;margin:0 0 8px 0;",
            ),
            *cards,
        )

    @output
    @render.table
    def prediction_advanced_table():
        if input.player_type() != "pitcher" or input.data_source() not in ("trackman",):
            return pd.DataFrame()
        st = ml_prediction_state()
        if not st.get("use_ml") or st.get("df") is None or st.get("df").empty:
            return pd.DataFrame()
        adf = st["df"].copy()
        keep = [PITCH_TYPE_COL, "attack_score", "putaway_score", "danger_score", "composite_score"]
        keep = [c for c in keep if c in adf.columns]
        adf = adf.sort_values("composite_score", ascending=False, kind="mergesort")
        adf = adf[keep].rename(columns={
            PITCH_TYPE_COL: "Pitch",
            "attack_score": "Control impact",
            "putaway_score": "Two-strike put-away impact",
            "danger_score": "Hard-contact risk impact",
            "composite_score": "Overall profile impact",
        })
        for c in ["Control impact", "Two-strike put-away impact", "Hard-contact risk impact", "Overall profile impact"]:
            if c in adf.columns:
                adf[c] = adf[c].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "—")
        return adf

    # -----------------------------------------------------------------------
    # Comparison tab reactives
    # -----------------------------------------------------------------------
    @reactive.effect
    def _update_cmp_opponent_teams():
        df = current_df()
        if df is None or df.empty:
            try:
                ui.update_select("cmp_opponent_team", choices={}, session=session)
            except Exception:
                pass
            return

        ptype = input.player_type()
        if ptype == "batter":
            col = "BatterTeam"
        else:
            col = "PitcherTeam"

        if col not in df.columns:
            try:
                ui.update_select("cmp_opponent_team", choices={}, session=session)
            except Exception:
                pass
            return

        teams = sorted([
            t for t in df[col].dropna().astype(str).str.strip().unique().tolist()
            if t
        ])

        choices = {t: TEAM_NAME_MAP.get(t, t) for t in teams}
        choices = dict(sorted(choices.items(), key=lambda x: x[1]))

        default_team = list(choices.keys())[0] if choices else None

        try:
            ui.update_select(
                "cmp_opponent_team",
                choices=choices,
                selected=default_team,
                session=session,
            )
        except Exception:
            pass

        # Also populate the opponent player dropdown immediately for the default team
        # (avoids timing issue where player dropdown is empty on first tab click)
        if default_team and not df.empty:
            ptype = input.player_type()
            if ptype == "pitcher":
                _populate_cmp_pitcher(df, default_team)
            else:
                _populate_cmp_batter(df, default_team)

    def _populate_cmp_pitcher(df, opp_team):
        d = df[df["PitcherTeam"].astype(str).str.strip() == str(opp_team)]
        if d.empty or "PitcherId" not in d.columns or "Pitcher" not in d.columns:
            try: ui.update_select("cmp_opponent_pitcher", choices={}, session=session)
            except: pass
            return
        lookup = d[["PitcherId","Pitcher"]].dropna().drop_duplicates().sort_values(["Pitcher","PitcherId"])
        choices = {str(r.PitcherId): (format_display_name(r.Pitcher) or str(r.PitcherId)) for r in lookup.itertuples(index=False)}
        if str(opp_team) == PURDUE_CODE:
            pid1 = input.player()
            choices = {k: v for k, v in choices.items() if k != str(pid1)}
        try: ui.update_select("cmp_opponent_pitcher", choices=choices, selected=list(choices.keys())[0] if choices else None, session=session)
        except: pass

    def _populate_cmp_batter(df, opp_team):
        d = df[df["BatterTeam"].astype(str).str.strip() == str(opp_team)]
        if d.empty or "BatterId" not in d.columns or "Batter" not in d.columns:
            try: ui.update_select("cmp_opponent_batter", choices={}, session=session)
            except: pass
            return
        lookup = d[["BatterId","Batter"]].dropna().drop_duplicates().sort_values(["Batter","BatterId"])
        choices = {str(r.BatterId): (format_display_name(r.Batter) or str(r.BatterId)) for r in lookup.itertuples(index=False)}
        if str(opp_team) == PURDUE_CODE:
            bid1 = input.player()
            choices = {k: v for k, v in choices.items() if k != str(bid1)}
        try: ui.update_select("cmp_opponent_batter", choices=choices, selected=list(choices.keys())[0] if choices else None, session=session)
        except: pass

    @reactive.effect
    def _update_cmp_opponent_pitchers():
        # Read extra dependencies to ensure this re-fires on filter changes
        _ = input.team()
        _ = input.data_source()
        if input.player_type() == "batter":
            return  # batter mode uses _update_cmp_opponent_batters instead
        df = current_df()
        opp_team = input.cmp_opponent_team()

        if df is None or df.empty or not opp_team:
            try:
                ui.update_select("cmp_opponent_pitcher", choices={}, session=session)
            except Exception:
                pass
            return

        d = df[df["PitcherTeam"].astype(str).str.strip() == str(opp_team)]

        if d.empty or "PitcherId" not in d.columns or "Pitcher" not in d.columns:
            try:
                ui.update_select("cmp_opponent_pitcher", choices={}, session=session)
            except Exception:
                pass
            return

        lookup = (
            d[["PitcherId", "Pitcher"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["Pitcher", "PitcherId"])
        )

        choices = {
            str(r.PitcherId): (format_display_name(r.Pitcher) or str(r.PitcherId))
            for r in lookup.itertuples(index=False)
        }

        if str(opp_team) == PURDUE_CODE:
            pid1 = input.player()
            choices = {k: v for k, v in choices.items() if k != str(pid1)}

        try:
            ui.update_select(
                "cmp_opponent_pitcher",
                choices=choices,
                selected=list(choices.keys())[0] if choices else None,
                session=session,
            )
        except Exception:
            pass


    @reactive.calc
    def cmp_purdue_df():
        df = current_df()
        pid = input.player()

        if df is None or df.empty or not pid:
            return pd.DataFrame()

        d = df[
            (df["PitcherTeam"].astype(str).str.strip() == PURDUE_CODE) &
            (df["PitcherId"].astype(str) == str(pid))
        ].copy()

        d = apply_session_filter_for_team(d, PURDUE_CODE, input.session_type())
        return d

    @reactive.calc
    def cmp_opponent_df():
        df = current_df()
        opp_team = input.cmp_opponent_team()
        opp_pid = input.cmp_opponent_pitcher()

        if df is None or df.empty or not opp_team or not opp_pid:
            return pd.DataFrame()

        d = df[
            (df["PitcherTeam"].astype(str).str.strip() == str(opp_team)) &
            (df["PitcherId"].astype(str) == str(opp_pid))
        ].copy()
        return d

    

    @reactive.effect
    def _update_cmp_pitch_type_choices():
        pur = cmp_purdue_df()
        opp = cmp_opponent_df()

        if pur is None or pur.empty or opp is None or opp.empty:
            ui.update_select(
                "cmp_pitch_type",
                choices={"all": "All Pitches"},
                selected="all",
                session=session,
            )
            return

        pur_types = set(
            pur[PITCH_TYPE_COL].dropna().astype(str).str.strip().tolist()
        ) if PITCH_TYPE_COL in pur.columns else set()

        opp_types = set(
            opp[PITCH_TYPE_COL].dropna().astype(str).str.strip().tolist()
        ) if PITCH_TYPE_COL in opp.columns else set()

        shared = sorted([pt for pt in (pur_types & opp_types) if pt])

        choices = {"all": "All Pitches"}
        for pt in shared:
            choices[pt] = pt

        selected = input.cmp_pitch_type()
        if selected not in choices:
            selected = "all"

        ui.update_select(
            "cmp_pitch_type",
            choices=choices,
            selected=selected,
            session=session,
        )

    @reactive.calc
    def cmp_purdue_filtered():
        return filter_df_to_pitch_type(cmp_purdue_df(), input.cmp_pitch_type())

    @reactive.calc
    def cmp_opponent_filtered():
        return filter_df_to_pitch_type(cmp_opponent_df(), input.cmp_pitch_type())

    @reactive.calc
    def cmp_purdue_name():
        d = cmp_purdue_df()
        if d is None or d.empty:
            return "Purdue Pitcher"
        return format_display_name(d["Pitcher"].iloc[0]) or "Purdue Pitcher"

    @reactive.calc
    def cmp_opponent_name():
        d = cmp_opponent_df()
        if d is None or d.empty:
            return "Opponent Pitcher"
        return format_display_name(d["Pitcher"].iloc[0]) or "Opponent Pitcher"

    @reactive.calc
    def cmp_pitch_type_label():
        val = input.cmp_pitch_type()
        return "All Pitches" if val in (None, "", "all") else str(val)


    # ── Batter comparison reactives ─────────────────────────────────────────
    @reactive.effect
    def _update_cmp_opponent_batters():
        # Read extra dependencies to ensure this re-fires on filter changes
        _ = input.team()
        _ = input.data_source()
        if input.player_type() != "batter":
            return
        df = current_df()
        opp_team = input.cmp_opponent_team()

        if df is None or df.empty or not opp_team:
            try:
                ui.update_select("cmp_opponent_batter", choices={}, session=session)
            except Exception:
                pass
            return

        d = df[df["BatterTeam"].astype(str).str.strip() == str(opp_team)]

        if d.empty or "BatterId" not in d.columns or "Batter" not in d.columns:
            try:
                ui.update_select("cmp_opponent_batter", choices={}, session=session)
            except Exception:
                pass
            return

        lookup = (
            d[["BatterId", "Batter"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["Batter", "BatterId"])
        )

        choices = {
            str(r.BatterId): (format_display_name(r.Batter) or str(r.BatterId))
            for r in lookup.itertuples(index=False)
        }

        if str(opp_team) == PURDUE_CODE:
            bid1 = input.player()
            choices = {k: v for k, v in choices.items() if k != str(bid1)}

        try:
            ui.update_select(
                "cmp_opponent_batter",
                choices=choices,
                selected=list(choices.keys())[0] if choices else None,
                session=session,
            )
        except Exception:
            pass

    @reactive.calc
    def cmp_batter_purdue_df():
        df = current_df()
        bid = input.player() if input.player_type() == "batter" else None
        if df is None or df.empty or not bid:
            return pd.DataFrame()
        team = input.team()
        if team and "BatterTeam" in df.columns:
            df = df[df["BatterTeam"].astype(str).str.strip() == str(team)]
        df = apply_session_filter_for_team(df, team, input.session_type())
        df = df[df["BatterId"].astype(str) == str(bid)]
        return df

    @reactive.calc
    def cmp_batter_opponent_df():
        df = current_df()
        if df is None or df.empty:
            return pd.DataFrame()
        try:
            opp_team = input.cmp_opponent_team()
            opp_bid = input.cmp_opponent_batter()
        except Exception:
            return pd.DataFrame()
        if not opp_team or not opp_bid:
            return pd.DataFrame()
        d = df[df["BatterTeam"].astype(str).str.strip() == str(opp_team)]
        return d[d["BatterId"].astype(str) == str(opp_bid)]

    @reactive.calc
    def cmp_batter_purdue_name():
        d = cmp_batter_purdue_df()
        if d.empty or "Batter" not in d.columns:
            return "Purdue Batter"
        return format_display_name(d["Batter"].iloc[0]) or "Purdue Batter"

    @reactive.calc
    def cmp_batter_opponent_name():
        d = cmp_batter_opponent_df()
        if d.empty or "Batter" not in d.columns:
            return "Opponent Batter"
        return format_display_name(d["Batter"].iloc[0]) or "Opponent Batter"

    # ── Batter comparison render functions ───────────────────────────────────
    @output
    @render.ui
    def cmp_batter_batting_line():
        pur_df = cmp_batter_purdue_df()
        opp_df = cmp_batter_opponent_df()
        pur_name = cmp_batter_purdue_name()
        opp_name = cmp_batter_opponent_name()

        bid_pur = input.player() if input.player_type() == "batter" else None
        try:
            bid_opp = input.cmp_opponent_batter()
        except Exception:
            bid_opp = None

        pur_s = compute_batter_stats(pur_df, bid_pur) if bid_pur else {}
        opp_s = compute_batter_stats(opp_df, bid_opp) if bid_opp else {}

        def _get_side(df):
            if df is None or df.empty or "BatterSide" not in df.columns:
                return ""
            side = str(df["BatterSide"].iloc[0]).strip()
            if side.lower() in ("nan", ""):
                return ""
            return side

        pur_side = _get_side(pur_df)
        opp_side = _get_side(opp_df)

        def fmt(v):
            if v is None:
                return "—"
            if v >= 1.0:
                return f"{v:.3f}"
            return f".{int(round(v * 1000)):03d}"

        metrics = ["PA", "BA", "OBP", "SLG", "OPS", "K", "BB"]
        keys = ["PA", "BA", "OBP", "SLG", "OPS", "K", "BB"]

        th_style = (
            "padding:8px 12px;text-align:center;font-weight:700;font-size:11px;"
            "background:#1a1a1a;color:#CDA735;"
        )
        td_style = "padding:8px 12px;text-align:center;font-size:13px;border-bottom:1px solid #eee;color:#333;font-weight:700;"

        def _get_val(s, key):
            if not s:
                return "—"
            v = s.get(key)
            if v is None:
                return "—"
            if key in ("BA", "OBP", "SLG", "OPS"):
                return fmt(v)
            return str(v)

        def _card(name, side, s, color, df=None, bid=None):
            side_txt = f" | Bats: {side}" if side else ""
            header_style = (
                f"font-size:14px;font-weight:700;color:#fff;padding:10px 14px;"
                f"background:{color};border-radius:10px 10px 0 0;"
                "display:flex;align-items:center;gap:10px;"
            )
            rows = [ui.tags.td(_get_val(s, k), style=td_style) for k in keys]

            # Low sample warning badge (same style as pitcher comparison)
            n_pitches = 0
            if df is not None and not df.empty and bid:
                n_pitches = len(df[df["BatterId"].astype(str) == str(bid)])
            warn_badge = []
            if 0 < n_pitches < 20:
                warn_badge = [ui.tags.span(
                    "⚠ Low Sample",
                    style=(
                        "display:inline-block;background:#fef3c7;border:1px solid #f59e0b;"
                        "border-radius:12px;padding:2px 10px;font-size:11px;"
                        "font-weight:700;color:#92400e;"
                    ),
                )]

            return ui.tags.div(
                ui.tags.div(
                    ui.tags.span(f"{name}{side_txt}"),
                    *warn_badge,
                    style=header_style,
                ),
                ui.tags.table(
                    ui.tags.thead(ui.tags.tr(*[ui.tags.th(m, style=th_style) for m in metrics])),
                    ui.tags.tbody(ui.tags.tr(*rows)),
                    style="width:100%;border-collapse:collapse;",
                ),
                style="background:#fff;border:1px solid #e0e0e0;border-radius:10px;overflow:hidden;flex:1;",
            )

        return ui.div(
            _card(pur_name, pur_side, pur_s, "#CDA735", pur_df, bid_pur),
            _card(opp_name, opp_side, opp_s, "#555", opp_df, bid_opp),
            style="display:flex;gap:14px;margin-bottom:14px;",
        )

    def _cmp_batter_data_pair():
        """Get both batter DataFrames + names + IDs for comparison charts."""
        pur_df = cmp_batter_purdue_df()
        opp_df = cmp_batter_opponent_df()
        pur_name = cmp_batter_purdue_name()
        opp_name = cmp_batter_opponent_name()
        bid_pur = input.player() if input.player_type() == "batter" else None
        try: bid_opp = input.cmp_opponent_batter()
        except: bid_opp = None
        return pur_df, opp_df, pur_name, opp_name, bid_pur, bid_opp

    def _cmp_prep_batter(df, bid):
        """Filter + clean location data for one batter."""
        if df is None or df.empty or not bid:
            return None
        d = df[df["BatterId"].astype(str) == str(bid)].copy()
        for c in ["PlateLocSide", "PlateLocHeight"]:
            if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if "PitchCall" not in d.columns: d["PitchCall"] = ""
        if "PlayResult" not in d.columns: d["PlayResult"] = ""
        return d if not d.empty else None

    @output
    @render.plot
    def cmp_batter_out_rate():
        pur_df, opp_df, pur_name, opp_name, bid_pur, bid_opp = _cmp_batter_data_pair()

        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        fig.patch.set_facecolor("#f7f7f7")

        OUT_RESULTS = {"Out", "FieldersChoice", "Strikeout", "Sacrifice",
                       "SacrificeFly", "SacrificeBunt", "Error"}
        PA_RESULTS = {"Single", "Double", "Triple", "HomeRun", "Out", "FieldersChoice",
                      "Strikeout", "Walk", "HitByPitch", "SacrificeFly", "SacrificeBunt",
                      "Sacrifice", "Error", "CatcherInterference"}

        zw = (ZONE_RIGHT - ZONE_LEFT) / 3
        zh = (ZONE_TOP - ZONE_BOTTOM) / 3
        col_edges = [ZONE_LEFT + i * zw for i in range(4)]
        row_edges = [ZONE_BOTTOM + i * zh for i in range(4)]

        for idx, (df, bid, name) in enumerate([
            (pur_df, bid_pur, pur_name), (opp_df, bid_opp, opp_name),
        ]):
            ax = axes[idx]
            ax.set_facecolor("#f7f7f7")
            ax.set_title(name, fontsize=13, fontweight="bold", pad=15)

            d = _cmp_prep_batter(df, bid)
            if d is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#888")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
                continue

            pc = d["PitchCall"].astype(str).str.strip()
            pr = d["PlayResult"].astype(str).str.strip()
            is_terminal = pr.isin(PA_RESULTS) | pc.eq("HitByPitch")
            total_pa = 0; total_outs = 0

            ax.add_patch(Rectangle(
                (ZONE_LEFT - 0.35, ZONE_BOTTOM - 0.30),
                (ZONE_RIGHT - ZONE_LEFT) + 0.70, (ZONE_TOP - ZONE_BOTTOM) + 0.60,
                facecolor="#edf4ee", edgecolor="none", zorder=0,
            ))

            for ri in range(3):
                for ci in range(3):
                    x0, x1 = col_edges[ci], col_edges[ci + 1]
                    y0, y1 = row_edges[ri], row_edges[ri + 1]
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    in_cell = d["PlateLocSide"].between(x0, x1) & d["PlateLocHeight"].between(y0, y1)
                    term_in = in_cell & is_terminal
                    n_pa = int(term_in.sum())
                    n_outs = int((term_in & pr.isin(OUT_RESULTS)).sum())
                    total_pa += n_pa; total_outs += n_outs

                    if n_pa == 0:
                        cell_bg = "#f5f5f5"
                    else:
                        op = n_outs / n_pa
                        if op >= 0.75: cell_bg = "#c0392b"
                        elif op >= 0.50: cell_bg = "#f0b000"
                        elif op >= 0.25: cell_bg = "#f5dfb3"
                        else: cell_bg = "#ffffff"

                    ax.add_patch(Rectangle((x0, y0), zw, zh, facecolor=cell_bg, edgecolor="#aaa", linewidth=1, zorder=2))
                    if n_pa == 0:
                        ax.text(cx, cy, "—", ha="center", va="center", fontsize=10, color="#bbb", zorder=3)
                    else:
                        op = n_outs / n_pa
                        txt_c = "#fff" if op >= 0.50 else "#222"
                        ax.text(cx, cy + zh * 0.12, f"{int(round(op*100))}%",
                                ha="center", va="center", fontsize=12, fontweight="bold", color=txt_c, zorder=3)
                        ax.text(cx, cy - zh * 0.12, f"{n_outs}/{n_pa}",
                                ha="center", va="center", fontsize=8, color="#ffe0db" if op >= 0.75 else "#666", zorder=3)

            ax.add_patch(Rectangle(
                (ZONE_LEFT, ZONE_BOTTOM), ZONE_RIGHT - ZONE_LEFT, ZONE_TOP - ZONE_BOTTOM,
                fill=False, linewidth=2.5, edgecolor="#222", zorder=5,
            ))

            out_pct = f"{int(round(total_outs / total_pa * 100))}%" if total_pa > 0 else "—"
            ax.text(0.5, 1.00, f"Out Rate: {out_pct} | In-Zone PA: {total_pa}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#555")

            ax.set_xlim(ZONE_LEFT - 1.0, ZONE_RIGHT + 1.0)
            ax.set_ylim(ZONE_BOTTOM - 0.6, ZONE_TOP + 0.8)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

        fig.subplots_adjust(top=0.88, bottom=0.04, left=0.05, right=0.95, wspace=0.15)
        plt.close(fig)
        return fig

    @output
    @render.ui
    def cmp_batter_pitch_results():
        pur_df, opp_df, pur_name, opp_name, bid_pur, bid_opp = _cmp_batter_data_pair()

        RESULT_TRACES = [
            ("Swing & Miss", {"calls": {"StrikeSwinging"}, "symbol": "x", "color": "#E24B4A", "size": 10}),
            ("Foul", {"calls": {"FoulBallFieldable", "FoulBallNotFieldable"}, "symbol": "triangle-up", "color": "#7B68AE", "size": 9}),
            ("Called Strike", {"calls": {"StrikeCalled"}, "symbol": "square", "color": "#E67E22", "size": 8}),
            ("Ball", {"calls": {"BallCalled", "Ball"}, "symbol": "circle", "color": "#aaaaaa", "size": 7}),
        ]
        HIT_RESULTS = {"Single", "Double", "Triple", "HomeRun"}
        OUT_RESULTS = {"Out", "FieldersChoice", "Error", "Sacrifice", "SacrificeFly", "SacrificeBunt"}

        plotly_config = {
            "displayModeBar": True, "displaylogo": False, "scrollZoom": True,
            "modeBarButtonsToRemove": [
                "toImage", "select2d", "lasso2d",
                "hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines",
            ],
        }

        from plotly.subplots import make_subplots

        pur_d = _cmp_prep_batter(pur_df, bid_pur)
        opp_d = _cmp_prep_batter(opp_df, bid_opp)
        n_pur = len(pur_d) if pur_d is not None else 0
        n_opp = len(opp_d) if opp_d is not None else 0

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=[f"{pur_name} (n={n_pur})", f"{opp_name} (n={n_opp})"],
            horizontal_spacing=0.06,
        )

        legend_added = set()

        for col_idx, d in enumerate([pur_d, opp_d], start=1):
            fig.add_shape(type="rect",
                x0=ZONE_LEFT, y0=ZONE_BOTTOM, x1=ZONE_RIGHT, y1=ZONE_TOP,
                line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)",
                row=1, col=col_idx,
            )
            hp_x = [-0.35, -0.35, 0, 0.35, 0.35]
            hp_y = [1.2, 1.0, 0.9, 1.0, 1.2]
            fig.add_trace(go.Scatter(
                x=hp_x + [hp_x[0]], y=hp_y + [hp_y[0]],
                mode="lines", fill="toself", fillcolor="rgba(255,255,255,0.8)",
                line=dict(color="#555", width=1), showlegend=False, hoverinfo="skip",
            ), row=1, col=col_idx)

            if d is None:
                fig.add_annotation(text="No data", x=0, y=2.5, showarrow=False,
                                   font=dict(size=14, color="#888"), row=1, col=col_idx)
                continue

            pc = d["PitchCall"].astype(str).str.strip()

            for label, sty in RESULT_TRACES:
                mask = pc.isin(sty["calls"])
                if not mask.any(): continue
                g = d[mask]
                show = label not in legend_added
                fig.add_trace(go.Scatter(
                    x=g["PlateLocSide"], y=g["PlateLocHeight"],
                    mode="markers", name=label, showlegend=show,
                    legendgroup=label,
                    hovertemplate=f"{label}<br>Side: %{{x:.2f}}<br>Height: %{{y:.2f}}<extra></extra>",
                    marker=dict(size=sty["size"], color=sty["color"], symbol=sty["symbol"],
                                opacity=0.82, line=dict(color="white", width=0.5)),
                ), row=1, col=col_idx)
                legend_added.add(label)

            in_play = d[pc.eq("InPlay")]
            if not in_play.empty:
                ip_pr = in_play["PlayResult"].astype(str).str.strip()
                hits = in_play[ip_pr.isin(HIT_RESULTS)]
                outs = in_play[ip_pr.isin(OUT_RESULTS)]
                if not hits.empty:
                    show = "In-Play Hit" not in legend_added
                    fig.add_trace(go.Scatter(
                        x=hits["PlateLocSide"], y=hits["PlateLocHeight"],
                        mode="markers", name="In-Play Hit", showlegend=show,
                        legendgroup="In-Play Hit",
                        hovertemplate="Hit (%{text})<br>Side: %{x:.2f}<br>Height: %{y:.2f}<extra></extra>",
                        text=hits["PlayResult"].astype(str).str.strip(),
                        marker=dict(size=10, color="#2E8B57", symbol="diamond", opacity=0.85,
                                    line=dict(color="white", width=0.5)),
                    ), row=1, col=col_idx)
                    legend_added.add("In-Play Hit")
                if not outs.empty:
                    show = "In-Play Out" not in legend_added
                    fig.add_trace(go.Scatter(
                        x=outs["PlateLocSide"], y=outs["PlateLocHeight"],
                        mode="markers", name="In-Play Out", showlegend=show,
                        legendgroup="In-Play Out",
                        hovertemplate="Out<br>Side: %{x:.2f}<br>Height: %{y:.2f}<extra></extra>",
                        marker=dict(size=10, color="#8B0000", symbol="diamond", opacity=0.85,
                                    line=dict(color="white", width=0.5)),
                    ), row=1, col=col_idx)
                    legend_added.add("In-Play Out")

        # Independent axes per subplot
        fig.update_xaxes(range=[ZONE_LEFT-1.0, ZONE_RIGHT+1.0], showgrid=True,
                         gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                         showticklabels=False, row=1, col=1)
        fig.update_yaxes(range=[ZONE_BOTTOM-0.6, ZONE_TOP+0.8], showgrid=True,
                         gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                         scaleanchor="x", scaleratio=1,
                         showticklabels=False, row=1, col=1)
        fig.update_xaxes(range=[ZONE_LEFT-1.0, ZONE_RIGHT+1.0], showgrid=True,
                         gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                         showticklabels=False, row=1, col=2)
        fig.update_yaxes(range=[ZONE_BOTTOM-0.6, ZONE_TOP+0.8], showgrid=True,
                         gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                         scaleanchor="x2", scaleratio=1,
                         showticklabels=False, row=1, col=2)

        fig.update_layout(
            paper_bgcolor="#f7f7f7", plot_bgcolor="#f7f7f7",
            height=550, dragmode="zoom",
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5,
                font=dict(size=9), bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#ccc", borderwidth=1,
                itemclick="toggleothers", itemdoubleclick="toggle",
            ),
        )

        return ui.HTML(fig.to_html(
            full_html=False, include_plotlyjs=False,
            config=plotly_config,
        ))

    @output
    @render.plot
    def cmp_batter_radar():
        pur_df = cmp_batter_purdue_df()
        opp_df = cmp_batter_opponent_df()
        pur_name = cmp_batter_purdue_name()
        opp_name = cmp_batter_opponent_name()
        bid_pur = input.player() if input.player_type() == "batter" else None
        try:
            bid_opp = input.cmp_opponent_batter()
        except Exception:
            bid_opp = None

        LABELS = ["Zone\nSw%", "Contact%", "O-Contact%", "Chase%", "Whiff%", "Zone%"]
        N = len(LABELS)
        angles = [n / N * 2 * np.pi for n in range(N)] + [0]

        SW = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        CT = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}

        def _calc_values(df, bid):
            values = [0.0] * N
            if df is None or df.empty or not bid:
                return values
            d = df[df["BatterId"].astype(str) == str(bid)].copy()
            if d.empty:
                return values
            pc = d["PitchCall"].astype(str).str.strip() if "PitchCall" in d.columns else pd.Series("", index=d.index)
            if "PlateLocSide" in d.columns and "PlateLocHeight" in d.columns:
                d["PlateLocSide"] = pd.to_numeric(d["PlateLocSide"], errors="coerce")
                d["PlateLocHeight"] = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
                iz = d["PlateLocSide"].between(ZONE_LEFT, ZONE_RIGHT) & d["PlateLocHeight"].between(ZONE_BOTTOM, ZONE_TOP)
                oz = ~iz & d["PlateLocSide"].notna() & d["PlateLocHeight"].notna()
                n_iz = iz.sum()
                n_oz = oz.sum()
                iz_sw = (iz & pc.isin(SW)).sum()
                oz_sw = (oz & pc.isin(SW)).sum()
                oz_ct = (oz & pc.isin(CT)).sum()
                values[0] = iz_sw / n_iz if n_iz > 0 else 0
                values[3] = oz_sw / n_oz if n_oz > 0 else 0
                values[5] = n_iz / len(d) if len(d) > 0 else 0
                if oz_sw > 0:
                    values[2] = oz_ct / oz_sw
            total_sw = pc.isin(SW).sum()
            total_ct = pc.isin(CT).sum()
            total_whiff = (pc == "StrikeSwinging").sum()
            values[1] = total_ct / total_sw if total_sw > 0 else 0
            values[4] = total_whiff / total_sw if total_sw > 0 else 0
            return values

        pur_vals = _calc_values(pur_df, bid_pur)
        opp_vals = _calc_values(opp_df, bid_opp)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#f7f7f7")

        for ax, vals, name, color in [
            (ax1, pur_vals, pur_name, "#CDA735"),
            (ax2, opp_vals, opp_name, "#555"),
        ]:
            ax.set_facecolor("#f7f7f7")
            ax.set_title(name, fontsize=13, fontweight="bold", pad=15)

            # Dashed grid circles
            for r in [0.25, 0.50, 0.75, 1.0]:
                ax.plot(angles, [r] * (N + 1), color="#ddd", lw=0.6, ls="--", zorder=1)

            vals_closed = vals + [vals[0]]
            ax.plot(angles, vals_closed, color=color, linewidth=2.5, zorder=3)
            ax.fill(angles, vals_closed, color=color, alpha=0.18, zorder=2)

            # Data point dots
            ax.scatter(angles[:-1], vals, s=42, color=color, zorder=4, edgecolors="white", lw=0.8)

            # Value labels — position inside the shape to avoid overlap with axis labels
            for ang, val in zip(angles[:-1], vals):
                # Place label slightly inside the data point (toward center)
                r_pos = max(val - 0.10, 0.05) if val >= 0.50 else min(val + 0.15, 1.10)
                ax.text(ang, r_pos, f"{val * 100:.0f}%",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color=color, zorder=5)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(LABELS, fontsize=8)
            ax.tick_params(axis="x", pad=12)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.25, 0.50, 0.75])
            ax.set_yticklabels(["25%", "50%", "75%"], fontsize=7, color="#aaa")
            ax.spines["polar"].set_visible(False)

        fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.05, wspace=0.30)
        plt.close(fig)
        return fig

    @output
    @render.ui
    def cmp_summary_cards():
        pur = cmp_purdue_filtered()
        opp = cmp_opponent_filtered()

        if pur is None or pur.empty or opp is None or opp.empty:
            return ui.div("No comparison data available for the selected filters.")

        pur_all = cmp_purdue_df()
        opp_all = cmp_opponent_df()

        pur_hand = throws_to_short(pur_all["PitcherThrows"].iloc[0]) if "PitcherThrows" in pur_all.columns and not pur_all.empty else ""
        opp_hand = throws_to_short(opp_all["PitcherThrows"].iloc[0]) if "PitcherThrows" in opp_all.columns and not opp_all.empty else ""

        pur_metrics = compute_comparison_metrics(pur)
        opp_metrics = compute_comparison_metrics(opp)

        pitch_label = cmp_pitch_type_label()

        pur_card = ui.div(
            ui.div(
                ui.div(get_pitcher_team_logo_text(PURDUE_CODE), class_="cmp-card-logo"),
                ui.div(
                    ui.div(cmp_purdue_name(), class_="cmp-card-name"),
                    ui.div(f"{pur_hand} Pitcher" if pur_hand else "Pitcher", class_="cmp-card-subtitle"),
                ),
                class_="cmp-card-header",
            ),
            ui.div(
                ui.div(pitch_label, class_="cmp-chip"),
                ui.div(f"Pitch Count: {pur_metrics['pitch_count']}", class_="cmp-chip"),
                *([ ui.div("⚠ Low Sample", style="display:inline-block; background:#fef3c7; border:1px solid #f59e0b; border-radius:12px; padding:2px 10px; font-size:11px; font-weight:700; color:#92400e;") ] if pur_metrics["pitch_count"] < 50 else []),
                class_="cmp-chip-row",
            ),
            ui.div(
                ui.div(
                    ui.div("Avg Velo", class_="cmp-stat-label"),
                    ui.div(format_num(pur_metrics["avg_velo"]), class_="cmp-stat-value"),
                    class_="cmp-stat-box",
                ),
                ui.div(
                    ui.div("Max Velo", class_="cmp-stat-label"),
                    ui.div(format_num(pur_metrics["max_velo"]), class_="cmp-stat-value"),
                    class_="cmp-stat-box",
                ),
                
                ui.div(
                    ui.div("Strike %", class_="cmp-stat-label"),
                    ui.div(format_pct(pur_metrics["strike_pct"]), class_="cmp-stat-value"),
                    class_="cmp-stat-box",
                ),
                ui.div(
                    ui.div("Whiff %", class_="cmp-stat-label"),
                    ui.div(format_pct(pur_metrics["whiff_pct"]), class_="cmp-stat-value"),
                    class_="cmp-stat-box",
                ),
                class_="cmp-stat-strip",
            ),
            class_="cmp-card",
        )

        opp_team = input.cmp_opponent_team()
        opp_card = ui.div(
            ui.div(
                ui.div(get_pitcher_team_logo_text(opp_team), class_="cmp-card-logo"),
                ui.div(
                    ui.div(cmp_opponent_name(), class_="cmp-card-name"),
                    ui.div(f"{opp_hand} Pitcher" if opp_hand else "Pitcher", class_="cmp-card-subtitle"),
                ),
                class_="cmp-card-header",
            ),
            ui.div(
                ui.div(pitch_label, class_="cmp-chip"),
                ui.div(f"Pitch Count: {opp_metrics['pitch_count']}", class_="cmp-chip"),
                *([ ui.div("⚠ Low Sample", style="display:inline-block; background:#fef3c7; border:1px solid #f59e0b; border-radius:12px; padding:2px 10px; font-size:11px; font-weight:700; color:#92400e;") ] if opp_metrics["pitch_count"] < 50 else []),
                class_="cmp-chip-row",
            ),
            ui.div(
                ui.div(
                    ui.div("Avg Velo", class_="cmp-stat-label"),
                    ui.div(format_num(opp_metrics["avg_velo"]), class_="cmp-stat-value"),
                    class_="cmp-stat-box",
                ),

                ui.div(
                    ui.div("Max Velo", class_="cmp-stat-label"),
                    ui.div(format_num(opp_metrics["max_velo"]), class_="cmp-stat-value"),
                    class_="cmp-stat-box",
                ),

                ui.div(
                    ui.div("Strike %", class_="cmp-stat-label"),
                    ui.div(format_pct(opp_metrics["strike_pct"]), class_="cmp-stat-value"),
                    class_="cmp-stat-box",
                ),
                ui.div(
                    ui.div("Whiff %", class_="cmp-stat-label"),
                    ui.div(format_pct(opp_metrics["whiff_pct"]), class_="cmp-stat-value"),
                    class_="cmp-stat-box",
                ),
                class_="cmp-stat-strip",
            ),
            class_="cmp-card",
        )

        return ui.div(
            pur_card,
            opp_card,
            class_="cmp-summary-grid",
        )

    @output
    @render.table
    def cmp_table():
        pur = cmp_purdue_filtered()
        opp = cmp_opponent_filtered()

        if pur is None or pur.empty or opp is None or opp.empty:
            return pd.DataFrame(columns=["Metric", "Purdue", "Opponent"])

        pm = compute_comparison_metrics(pur)
        om = compute_comparison_metrics(opp)

        pur_col = f"Purdue\n"
        opp_col = f"Opponent\n"

        table = pd.DataFrame({
            "Metric": [
                "Velocity STDEV",
                "IVB Avg (in)",
                "HB Avg (in)",
            ],
            pur_col: [
                format_num(pm["velo_sd"]),
                format_num(pm["ivb_avg"]),
                format_num(pm["hb_avg"]),
            ],
            opp_col: [
                format_num(om["velo_sd"]),
                format_num(om["ivb_avg"]),
                format_num(om["hb_avg"]),
            ],
        })

        return table

    @output
    @render.ui
    def movement():
        mov = pitcher_mov_data()
        colors = pitch_colors()
        sel = selected_pitch.get()

        fig = go.Figure()

        if mov is None or mov.empty:
            fig.add_annotation(
                text="No pitch movement data",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14, color="#555555"),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(
                paper_bgcolor="#f7f7f7",
                plot_bgcolor="#f7f7f7",
                margin=dict(l=20, r=20, t=20, b=20),
                height=340,
            )
            return ui.HTML(fig.to_html(
                full_html=False, include_plotlyjs=False,
                config={"displayModeBar": False},
            ))

        for pt, g in mov.groupby(PITCH_TYPE_COL):
            color = colors.get(pt, "#777777")
            is_selected = (sel == pt)

            gx = pd.to_numeric(g[X_MOV], errors="coerce")
            gy = pd.to_numeric(g[Y_MOV], errors="coerce")
            valid = gx.notna() & gy.notna()

            g_valid = g.loc[valid].copy()
            if g_valid.empty:
                continue

            g_plot = g_valid.sample(
                n=min(len(g_valid), 30),
                random_state=42
            )

            fig.add_trace(
                go.Scatter(
                    x=g_plot[X_MOV],
                    y=g_plot[Y_MOV],
                    mode="markers",
                    name=pt,
                    showlegend=False,
                    hovertemplate=(
                        f"{pt}<br>"
                        "HB: %{x:.1f}<br>"
                        "IVB: %{y:.1f}<extra></extra>"
                    ),
                    marker=dict(
                        size=12 if is_selected else 8,
                        color=color,
                        opacity=1.0 if (not sel or is_selected) else 0.15,
                        line=dict(
                            color="black" if (not sel or is_selected) else color,
                            width=1 if (not sel or is_selected) else 0
                        ),
                    ),
                )
            )

        fig.add_hline(y=0, line_width=1.0, line_color="#4f83b6", opacity=0.75)
        fig.add_vline(x=0, line_width=1.0, line_color="#4f83b6", opacity=0.75)

        fig.update_xaxes(
            title="Horizontal break (in)",
            range=[-20, 20],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.10)",
            zeroline=False,
        )

        fig.update_yaxes(
            title="Induced vertical break (in)",
            range=[-20, 20],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.10)",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        )

        fig.update_layout(
            paper_bgcolor="#f7f7f7",
            plot_bgcolor="#f7f7f7",
            margin=dict(l=50, r=20, t=20, b=45),
            height=340,
            dragmode="zoom",
        )

        return ui.HTML(fig.to_html(
            full_html=False, include_plotlyjs=False,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": [
                    "toImage", "select2d", "lasso2d",
                    "hoverClosestCartesian", "hoverCompareCartesian",
                    "toggleSpikelines",
                ],
            },
        ))

    @output
    @render.plot
    def cmp_location():
        pur = cmp_purdue_filtered()
        opp = cmp_opponent_filtered()

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
        fig.patch.set_facecolor("#f7f7f7")

        titles = ["Purdue", "Opponent"]
        data_list = [pur, opp]
        colors = ["#DDB945", "#9E9E9E"]

        for ax, title, d, c in zip(axes, titles, data_list, colors):
            ax.set_facecolor("#f7f7f7")

            if d is None or d.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            loc = d[
                d["PlateLocSide"].notna() &
                d["PlateLocHeight"].notna()
            ]

            if loc.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            # outer light zone (same as home chart)
            ax.add_patch(Rectangle(
                (ZONE_LEFT - 0.3, ZONE_BOTTOM - 0.3),
                (ZONE_RIGHT - ZONE_LEFT) + 0.6,
                (ZONE_TOP - ZONE_BOTTOM) + 0.6,
                alpha=0.25,
                facecolor="#d9d9d9",
                edgecolor="none"
            ))

            # blue attack zone (same visual size as home chart)
            ax.add_patch(Rectangle(
                (-1.1, 1.2),
                2.2,
                2.6,
                alpha=0.28,
                facecolor="#8fb7d6",
                edgecolor="none"
            ))

            # strike zone
            ax.add_patch(Rectangle(
                (ZONE_LEFT, ZONE_BOTTOM),
                ZONE_RIGHT - ZONE_LEFT,
                ZONE_TOP - ZONE_BOTTOM,
                fill=False,
                linewidth=2.4,
                edgecolor="black"
            ))

            # center reference lines (same colors as home)
            ax.plot(
                [ZONE_LEFT, ZONE_RIGHT],
                [(ZONE_BOTTOM + ZONE_TOP) / 2] * 2,
                linestyle="--",
                linewidth=1.4,
                color="#1f77b4"
            )

            ax.plot(
                [0, 0],
                [ZONE_BOTTOM, ZONE_TOP],
                linestyle="--",
                linewidth=1.4,
                color="#ff7f0e"
            )

            ax.scatter(
                loc["PlateLocSide"],
                loc["PlateLocHeight"],
                s=35,
                alpha=0.8,
                color=c,
                edgecolors="none"
            )

            ax.add_patch(home_plate_polygon(y_front=0.10))

            ax.set_xlim(-3, 3)
            ax.set_ylim(-0.5, 5)
            ax.set_aspect("equal", adjustable="box")

            ax.set_title(title, fontsize=12, fontweight="bold")

            ax.grid(True, alpha=0.2)

            ax.set_xticks([])
            ax.set_yticks([])

        plt.close(fig)
        return fig

    @output
    @render.plot
    def cmp_count_location():
        pur = cmp_purdue_filtered()
        opp = cmp_opponent_filtered()

        COUNTS = [
            (0, 0, "0-0", "FIRST PITCH"),
            (3, 0, "3-0", "HITTER'S COUNT"),
            (0, 2, "0-2", "PITCHER AHEAD"),
            (1, 2, "1-2", "PITCHER AHEAD"),
            (2, 2, "2-2", "EVEN COUNT"),
            (3, 2, "3-2", "FULL COUNT"),
        ]
        OUTCOME_COLORS = {
            "StrikeCalled":         "#22c55e",
            "StrikeSwinging":       "#ef4444",
            "Ball":                 "#475569",
            "BallCalled":           "#475569",   
            "FoulBallFieldable":    "#f97316",
            "FoulBallNotFieldable": "#f97316",
            "InPlay":               "#3b82f6",
        }
        DEFAULT_COLOR = "#cbd5e1"
        STRIKE_EVENTS = {
            "StrikeCalled", "StrikeSwinging",
            "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"
        }

        def _get_count_df(d, balls, strikes):
            if d is None or d.empty:
                return pd.DataFrame()
            b_col = _find_col(d, ["Balls", "balls"])
            s_col = _find_col(d, ["Strikes", "strikes"])
            if b_col is None or s_col is None:
                return pd.DataFrame()
            mask = (
                pd.to_numeric(d[b_col], errors="coerce").eq(balls) &
                pd.to_numeric(d[s_col], errors="coerce").eq(strikes)
            )
            return d.loc[mask]

        def _spct(df):
            if df is None or df.empty or "PitchCall" not in df.columns:
                return None
            n = len(df)
            if n == 0:
                return None
            return 100 * df["PitchCall"].astype(str).isin(STRIKE_EVENTS).sum() / n

        def _draw_zone(ax, df):
            ax.set_facecolor("white")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.55, 4.55)
            ax.set_clip_on(True)
            ax.patch.set_zorder(1)
            ax.add_patch(Rectangle(
                (-1.1, 1.2), 2.2, 2.6,
                alpha=0.20, facecolor="#bfdbfe", edgecolor="none", zorder=2, clip_on=True
            ))
            ax.add_patch(Rectangle(
                (ZONE_LEFT, ZONE_BOTTOM), ZONE_RIGHT - ZONE_LEFT, ZONE_TOP - ZONE_BOTTOM,
                fill=False, linewidth=2.0, edgecolor="#1e293b", zorder=2, clip_on=True
            ))

            mid_y = (ZONE_BOTTOM + ZONE_TOP) / 2
            ax.plot([ZONE_LEFT, ZONE_RIGHT], [mid_y, mid_y],
                    linestyle="--", linewidth=1.0, color="#60a5fa", alpha=0.6, zorder=1)
            ax.plot([0, 0], [ZONE_BOTTOM, ZONE_TOP],
                    linestyle="--", linewidth=1.0, color="#f97316", alpha=0.6, zorder=1)

            if df is not None and not df.empty:
                loc = df[df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()]
                if not loc.empty:
                    dot_colors = [
                        OUTCOME_COLORS.get(str(pc).strip(), DEFAULT_COLOR)
                        for pc in loc["PitchCall"].astype(str)
                    ]
                    ax.scatter(
                        loc["PlateLocSide"], loc["PlateLocHeight"],
                        s=70, alpha=0.90, color=dot_colors,
                        edgecolors="white", linewidths=0.35,
                        zorder=3, clip_on=True
                    )

            ax.add_patch(home_plate_polygon(y_front=0.10))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#c8d0da")
                spine.set_linewidth(1.2)

        def _draw_info(ax, count_label, count_context, pur_s, opp_s, n_pur, n_opp):
            ax.set_facecolor("white")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#e2e8f0")
                spine.set_linewidth(1.0)
            ax.text(0.5, 0.80, count_label,
                    ha="center", va="center",
                    fontsize=26, fontweight="black", color="#1e293b",
                    transform=ax.transAxes)
            ax.text(0.5, 0.63, count_context,
                    ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#64748b",
                    transform=ax.transAxes)
            ax.text(0.5, 0.49, "Strike Percentage",
                    ha="center", va="center",
                    fontsize=7, color="#94a3b8",
                    transform=ax.transAxes)
            pur_txt = f"{pur_s:.0f}%" if pur_s is not None else "—"
            opp_txt = f"{opp_s:.0f}%" if opp_s is not None else "—"
            ax.text(0.26, 0.33, pur_txt,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#111827",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.38",
                              facecolor="#DDB945", edgecolor="none"))
            ax.text(0.74, 0.33, opp_txt,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.38",
                              facecolor="#1e293b", edgecolor="none"))
            ax.text(0.26, 0.17, f"n={n_pur}",
                    ha="center", va="center",
                    fontsize=7, color="#94a3b8",
                    transform=ax.transAxes)
            ax.text(0.74, 0.17, f"n={n_opp}",
                    ha="center", va="center",
                    fontsize=7, color="#94a3b8",
                    transform=ax.transAxes)

        # ── build figure ───────────────────────────────────────────────────────
        pur_name = cmp_purdue_name() or "Purdue Pitcher"
        opp_name = cmp_opponent_name() or "Opponent Pitcher"
        n_rows   = len(COUNTS)
        BG       = "#e8edf2"

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(10.0, 3.2 * n_rows + 0.8), facecolor=BG)

        # Tiny legend row on top, data rows below
        gs_outer = GridSpec(2, 1, figure=fig,
                            height_ratios=[0.01, 0.96],
                            hspace=0.0)

        ax_leg = fig.add_subplot(gs_outer[0, 0])
        ax_leg.set_facecolor(BG)
        ax_leg.axis("off")
        legend_items = [
            ("Called Strike",   "#22c55e"),
            ("Swinging Strike", "#ef4444"),
            ("Ball",            "#475569"),
            ("Foul",            "#f97316"),
            ("In Play",         "#3b82f6"),
        ]
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=c, markersize=8, label=lbl)
            for lbl, c in legend_items
        ]
        leg = ax_leg.legend(
            handles=legend_handles,
            loc="center", ncol=5, fontsize=8.5,
            bbox_to_anchor=(0.5, 0.10),
            frameon=True, fancybox=False,
            edgecolor="#cbd5e1", facecolor="white",
            borderpad=0.5, handletextpad=0.4, columnspacing=0.9,
        )
        leg.get_frame().set_linewidth(0.8)

        # Data rows
        gs = GridSpec(
            nrows=n_rows, ncols=3,
            width_ratios=[0.55, 1.7, 1.7],
            height_ratios=[1] * n_rows,
            wspace=0.03, hspace=0.20,
            left=0.015, right=0.985,
            top=0.955, bottom=0.008,
        )

        row_axes     = []
        first_pur_ax = None
        first_opp_ax = None

        for row_idx, (balls, strikes, count_label, count_context) in enumerate(COUNTS):
            pur_cdf = _get_count_df(pur, balls, strikes)
            opp_cdf = _get_count_df(opp, balls, strikes)
            pur_s   = _spct(pur_cdf)
            opp_s   = _spct(opp_cdf)
            n_pur   = len(pur_cdf) if not pur_cdf.empty else 0
            n_opp   = len(opp_cdf) if not opp_cdf.empty else 0

            ax_info = fig.add_subplot(gs[row_idx, 0])
            ax_pur  = fig.add_subplot(gs[row_idx, 1])
            ax_opp  = fig.add_subplot(gs[row_idx, 2])

            for ax in [ax_info, ax_pur, ax_opp]:
                ax.set_facecolor("white")
                for spine in ax.spines.values():
                    spine.set_visible(False)

            _draw_info(ax_info, count_label, count_context,
                       pur_s, opp_s, n_pur, n_opp)
            _draw_zone(ax_pur, pur_cdf)
            _draw_zone(ax_opp, opp_cdf)

            row_axes.append((ax_info, ax_opp))
            if row_idx == 0:
                first_pur_ax = ax_pur
                first_opp_ax = ax_opp

        # Player names + card borders after layout
        fig.canvas.draw()

        pp = first_pur_ax.get_position()
        po = first_opp_ax.get_position()
        name_y = pp.y1 + 0.005
        fig.text((pp.x0 + pp.x1) / 2, name_y, pur_name,
                 ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#1e293b",
                 transform=fig.transFigure)
        fig.text((po.x0 + po.x1) / 2, name_y, opp_name,
                 ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#1e293b",
                 transform=fig.transFigure)

        PAD = 0.005
        for i, (ax_left, ax_right) in enumerate(row_axes):
            pl = ax_left.get_position()
            pr = ax_right.get_position()
            extra_top = 0.03 if i == 0 else 0.0   # extend first card to cover player name
            fig.add_artist(FancyBboxPatch(
                (pl.x0 - PAD, pl.y0 - PAD),
                (pr.x1 - pl.x0) + 2 * PAD,
                pl.height + 2 * PAD + extra_top,
                boxstyle="round,pad=0.003",
                facecolor="white", edgecolor="#c8d0da",
                linewidth=1.1, transform=fig.transFigure,
                zorder=0, clip_on=False,
            ))

        plt.close(fig)
        return fig

    
    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None


    def _safe_pct(num, den):
        if den is None or den == 0:
            return np.nan
        return 100.0 * num / den


    def _format_pct(x):
        if pd.isna(x):
            return "—"
        return f"{x:.1f}%"


    def _format_num(x, digits=0):
        if pd.isna(x):
            return "—"
        return f"{x:.{digits}f}"


    def _team_summary_metrics(df):
        if df is None or df.empty:
            return {
                "strike_pct": np.nan,
                "whiff_pct": np.nan,
                "chase_pct": np.nan,
                "first_pitch_strike_pct": np.nan,

            }

        d = df.copy()

        pitch_call_col = _find_col(d, ["PitchCall", "pitchCall"])
        play_result_col = _find_col(d, ["PlayResult", "playResult"])
        spin_col = _find_col(d, ["SpinRate", "RelSpeedSpinRate", "spin_rate"])
        pitch_no_col = _find_col(d, ["PitchofPA", "PitchOfPA", "pitch_number_pa", "PitchNoPA"])
        balls_col = _find_col(d, ["Balls", "balls"])
        strikes_col = _find_col(d, ["Strikes", "strikes"])
        plate_side_col = _find_col(d, ["PlateLocSide", "plate_x"])
        plate_height_col = _find_col(d, ["PlateLocHeight", "plate_z"])

        strike_calls = {
            "StrikeCalled",
            "StrikeSwinging",
            "FoulBallFieldable",
            "FoulBallNotFieldable",
            "InPlay"
        }

        swinging_miss_calls = {
            "StrikeSwinging"
        }

        swing_calls = {
            "StrikeSwinging",
            "FoulBallNotFieldable",
            "FoulBallFieldable",
            "InPlay"
        }

        if pitch_call_col is not None:
            pitch_call_series = d[pitch_call_col].astype(str)
            total_pitches = len(d)

            strike_pct = _safe_pct(pitch_call_series.isin(strike_calls).sum(), total_pitches)
            whiff_pct = _safe_pct(pitch_call_series.isin(swinging_miss_calls).sum(), pitch_call_series.isin(swing_calls).sum())
        else:
            strike_pct = np.nan
            whiff_pct = np.nan

        chase_pct = np.nan
        if pitch_call_col is not None and plate_side_col is not None and plate_height_col is not None:
            px = pd.to_numeric(d[plate_side_col], errors="coerce")
            pz = pd.to_numeric(d[plate_height_col], errors="coerce")

            has_location = px.notna() & pz.notna()

            is_outside = (
                has_location & (
                    (px < ZONE_LEFT)   |
                    (px > ZONE_RIGHT)  |
                    (pz < ZONE_BOTTOM) |
                    (pz > ZONE_TOP)
                )
            )

            swings = d[pitch_call_col].astype(str).isin(swing_calls)
            outside_count = is_outside.sum()

            chase_pct = _safe_pct((swings & is_outside).sum(), outside_count)

        first_pitch_strike_pct = np.nan

        if pitch_no_col is not None and pitch_call_col is not None:
            first_pitch_mask = pd.to_numeric(d[pitch_no_col], errors="coerce") == 1
            fp = d.loc[first_pitch_mask].copy()

            if not fp.empty:
                first_pitch_strike_pct = _safe_pct(
                    fp[pitch_call_col].astype(str).isin(strike_calls).sum(),
                    len(fp)
                )
        elif balls_col is not None and strikes_col is not None and pitch_call_col is not None:
            first_pitch_mask = (
                pd.to_numeric(d[balls_col], errors="coerce").fillna(-1).eq(0) &
                pd.to_numeric(d[strikes_col], errors="coerce").fillna(-1).eq(0)
            )
            fp = d.loc[first_pitch_mask].copy()

            if not fp.empty:
                first_pitch_strike_pct = _safe_pct(
                    fp[pitch_call_col].astype(str).isin(strike_calls).sum(),
                    len(fp)
                )

        spin_rate = np.nan
        if spin_col is not None:
            spin_rate = pd.to_numeric(d[spin_col], errors="coerce").mean()

        return {
            "strike_pct": strike_pct,
            "whiff_pct": whiff_pct,
            "chase_pct": chase_pct,
            "first_pitch_strike_pct": first_pitch_strike_pct,
            "spin_rate": spin_rate,
            "ncaa_percentile": "—"
        }


    @output
    @output
    @output
    @render.ui
    def cmp_batter_takeaways():
        pur_df = cmp_batter_purdue_df()
        opp_df = cmp_batter_opponent_df()
        pur_name = cmp_batter_purdue_name()
        opp_name = cmp_batter_opponent_name()
        bid_pur = input.player() if input.player_type() == "batter" else None
        try:
            bid_opp = input.cmp_opponent_batter()
        except Exception:
            bid_opp = None

        pur_s = compute_batter_stats(pur_df, bid_pur) if bid_pur else {}
        opp_s = compute_batter_stats(opp_df, bid_opp) if bid_opp else {}

        if not pur_s or not opp_s or not pur_s.get("PA") or not opp_s.get("PA"):
            return ui.div("Select both batters to see takeaways.",
                          style="font-size:12px;color:#888;padding:12px;font-style:italic;")

        SW = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        CT = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}

        def _disc(df, bid):
            if df is None or df.empty or not bid:
                return {}
            d = df[df["BatterId"].astype(str) == str(bid)].copy()
            if d.empty:
                return {}
            pc = d["PitchCall"].astype(str).str.strip() if "PitchCall" in d.columns else pd.Series("", index=d.index)
            total_sw = pc.isin(SW).sum()
            total_whiff = (pc == "StrikeSwinging").sum()
            total_ct = pc.isin(CT).sum()
            result = {
                "whiff": total_whiff / total_sw if total_sw > 0 else 0,
                "contact": total_ct / total_sw if total_sw > 0 else 0,
            }
            if "PlateLocSide" in d.columns and "PlateLocHeight" in d.columns:
                lh = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
                ls = pd.to_numeric(d["PlateLocSide"], errors="coerce")
                iz = lh.between(ZONE_BOTTOM, ZONE_TOP) & ls.between(ZONE_LEFT, ZONE_RIGHT)
                oz = ~iz & lh.notna() & ls.notna()
                n_iz = iz.sum()
                n_oz = oz.sum()
                result["zone_sw"] = (iz & pc.isin(SW)).sum() / n_iz if n_iz > 0 else 0
                result["chase"] = (oz & pc.isin(SW)).sum() / n_oz if n_oz > 0 else 0
            return result

        pur_d = _disc(pur_df, bid_pur)
        opp_d = _disc(opp_df, bid_opp)

        insights = []

        # OPS comparison
        pur_ops = pur_s.get("OPS")
        opp_ops = opp_s.get("OPS")
        if pur_ops is not None and opp_ops is not None:
            if pur_ops > opp_ops + 0.05:
                insights.append(f"{pur_name} is more productive overall "
                                f"({pur_ops:.3f} OPS vs {opp_ops:.3f})")
            elif opp_ops > pur_ops + 0.05:
                insights.append(f"{opp_name} is more productive overall "
                                f"({opp_ops:.3f} OPS vs {pur_ops:.3f})")
            else:
                insights.append(f"Similar overall production — "
                                f"{pur_name} {pur_ops:.3f} OPS vs {opp_name} {opp_ops:.3f}")

        # BA comparison
        pur_ba = pur_s.get("BA")
        opp_ba = opp_s.get("BA")
        if pur_ba is not None and opp_ba is not None and abs(pur_ba - opp_ba) > 0.03:
            better = pur_name if pur_ba > opp_ba else opp_name
            higher = max(pur_ba, opp_ba)
            lower = min(pur_ba, opp_ba)
            insights.append(f"{better} has the higher batting average "
                            f"(.{int(round(higher*1000)):03d} vs .{int(round(lower*1000)):03d})")

        # Discipline comparison
        if pur_d and opp_d:
            pur_chase = pur_d.get("chase", 0)
            opp_chase = opp_d.get("chase", 0)
            if abs(pur_chase - opp_chase) > 0.05:
                more_disc = pur_name if pur_chase < opp_chase else opp_name
                less_disc = opp_name if pur_chase < opp_chase else pur_name
                insights.append(f"{more_disc} is more disciplined — "
                                f"chases {min(pur_chase,opp_chase):.0%} vs {max(pur_chase,opp_chase):.0%}")

            pur_whiff = pur_d.get("whiff", 0)
            opp_whiff = opp_d.get("whiff", 0)
            if abs(pur_whiff - opp_whiff) > 0.05:
                better_contact = pur_name if pur_whiff < opp_whiff else opp_name
                insights.append(f"{better_contact} makes more contact — "
                                f"{min(pur_whiff,opp_whiff):.0%} whiff rate vs {max(pur_whiff,opp_whiff):.0%}")

        # K and BB comparison
        pur_k = pur_s.get("K", 0)
        opp_k = opp_s.get("K", 0)
        pur_bb = pur_s.get("BB", 0)
        opp_bb = opp_s.get("BB", 0)
        if pur_k + opp_k > 0 and abs(pur_k - opp_k) > 2:
            more_k = pur_name if pur_k > opp_k else opp_name
            insights.append(f"{more_k} strikes out more ({max(pur_k,opp_k)} K vs {min(pur_k,opp_k)})")

        if not insights:
            return ui.div("Not enough data to generate takeaways.",
                          style="font-size:12px;color:#888;padding:12px;font-style:italic;")

        items = []
        for insight in insights[:5]:
            items.append(ui.tags.div(
                ui.tags.span("•", style="color:#CDA735;font-weight:900;margin-right:8px;font-size:16px;"),
                insight,
                style="font-size:12px;line-height:1.6;color:#333;padding:4px 0;",
            ))

        return ui.div(*items, style="padding:12px 16px;")

    @output
    @render.ui
    def cmp_player_select():
        ptype = input.player_type()
        if ptype == "batter":
            return ui.input_select("cmp_opponent_batter", "Opponent Batter", choices={})
        else:
            return ui.TagList(
                ui.input_select("cmp_opponent_pitcher", "Opponent Pitcher", choices={}),
                ui.input_select(
                    "cmp_pitch_type",
                    "Pitch Type Comparison",
                    choices={"all": "All Pitches"},
                    selected="all",
                ),
            )

    @output
    @render.ui
    def cmp_content():
        ptype = input.player_type()

        if ptype == "batter":
            return ui.div(
                ui.output_ui("cmp_batter_batting_line"),
                ui.div(
                    ui.div("Out Rate Comparison", class_="team-summary-title"),
                    ui.output_plot("cmp_batter_out_rate", height="500px"),
                    class_="team-summary-wrap",
                ),
                ui.div(
                    ui.div("Pitch Results Comparison", class_="team-summary-title"),
                    ui.output_ui("cmp_batter_pitch_results"),
                    class_="team-summary-wrap",
                ),
                ui.div(
                    ui.div("Plate Discipline Comparison", class_="team-summary-title"),
                    ui.output_plot("cmp_batter_radar", height="450px"),
                    class_="team-summary-wrap",
                ),
                ui.div(
                    ui.div("Key Takeaways", class_="team-summary-title"),
                    ui.output_ui("cmp_batter_takeaways"),
                    class_="team-summary-wrap",
                ),
            )
        else:
            return ui.div(
                ui.output_ui("cmp_team_summary_table"),
                ui.output_ui("cmp_summary_cards"),
                ui.div(
                    ui.div("Location by Count", class_="team-summary-title"),
                    ui.output_plot("cmp_count_location", height="2600px"),
                    class_="team-summary-wrap",
                ),
            )

    @output
    @render.ui
    def cmp_team_summary_table():
        pur = cmp_purdue_df()
        opp = cmp_opponent_df()

        pur_m = _team_summary_metrics(pur)
        opp_m = _team_summary_metrics(opp)


        def _winner(pur_val, opp_val):
            """Returns 'purdue', 'opp', or None if tied / both missing."""
            if pd.isna(pur_val) or pd.isna(opp_val):
                return None
            pur_val = round(float(pur_val) * 100, 1)
            opp_val = round(float(opp_val) * 100, 1)
            if pur_val > opp_val:
                return "purdue"
            if opp_val > pur_val:
                return "opp"
            return None

        # ── helper: styled <td> based on whether this team wins the metric ──
        def _td(text, is_winner, is_loser):
            if is_winner:
                style = (
                    "color:#15803d; font-weight:900; background:#dcfce7;"
                    "border-radius:4px; padding:3px 10px; display:inline-block;"
                )
            elif is_loser:
                style = (
                    "color:#b91c1c; font-weight:700; background:#fee2e2;"
                    "border-radius:4px; padding:3px 10px; display:inline-block;"
                )
            else:
                style = "color:#64748b; padding:3px 10px; display:inline-block;"
            return ui.tags.td(ui.tags.span(text, style=style))

        # ── compute winners per metric ──
        metrics = [
            ("strike_pct",             _format_pct(pur_m["strike_pct"]),             _format_pct(opp_m["strike_pct"]),             _winner(pur_m["strike_pct"],             opp_m["strike_pct"])),
            ("whiff_pct",              _format_pct(pur_m["whiff_pct"]),              _format_pct(opp_m["whiff_pct"]),              _winner(pur_m["whiff_pct"],              opp_m["whiff_pct"])),
            ("chase_pct",              _format_pct(pur_m["chase_pct"]),              _format_pct(opp_m["chase_pct"]),              _winner(pur_m["chase_pct"],              opp_m["chase_pct"])),
            ("first_pitch_strike_pct", _format_pct(pur_m["first_pitch_strike_pct"]), _format_pct(opp_m["first_pitch_strike_pct"]), _winner(pur_m["first_pitch_strike_pct"], opp_m["first_pitch_strike_pct"])),
        ]

        # ── legend chips ──
        legend = ui.div(
            ui.tags.span(
                ui.tags.span("", style="display:inline-block;width:12px;height:12px;background:#dcfce7;border:1.5px solid #22c55e;border-radius:3px;margin-right:4px;vertical-align:middle;"),
                "Better",
                style="font-size:11px; color:#475569; margin-right:12px;"
            ),
            ui.tags.span(
                ui.tags.span("", style="display:inline-block;width:12px;height:12px;background:#fee2e2;border:1.5px solid #ef4444;border-radius:3px;margin-right:4px;vertical-align:middle;"),
                "Worse",
                style="font-size:11px; color:#475569;"
            ),
            style="display:flex; align-items:center; padding:6px 0 8px 0;"
        )

        # ── build rows ──
        pur_tds = [ui.tags.td("Purdue", class_="team-summary-purdue")]
        opp_tds = [ui.tags.td("Opponent", class_="team-summary-opponent")]

        for _, pur_val, opp_val, w in metrics:
            pur_tds.append(_td(pur_val, w == "purdue", w == "opp"))
            opp_tds.append(_td(opp_val, w == "opp",    w == "purdue"))

        # ── auto scout note ──  ← INSERT EVERYTHING FROM HERE
        purdue_wins = sum(1 for _, _, _, w in metrics if w == "purdue")
        opp_wins    = sum(1 for _, _, _, w in metrics if w == "opp")

        metric_labels = {
            "strike_pct":             "Strike %",
            "whiff_pct":              "Whiff %",
            "chase_pct":              "Chase %",
            "first_pitch_strike_pct": "First Pitch Strike %",
        }

        opp_threat_label, opp_threat_pur, opp_threat_opp = "", "", ""
        best_gap = -1
        for key, pur_val, opp_val, w in metrics:
            if w == "opp" and not pd.isna(pur_m[key]) and not pd.isna(opp_m[key]):
                gap = abs(opp_m[key] - pur_m[key])
                if gap > best_gap and round(gap * 100, 1) > 0:
                    best_gap = gap
                    opp_threat_label = metric_labels[key]
                    opp_threat_pur   = pur_val
                    opp_threat_opp   = opp_val

        pur_edge_label, pur_edge_pur, pur_edge_opp = "", "", ""
        best_gap = -1
        for key, pur_val, opp_val, w in metrics:
            if w == "purdue" and not pd.isna(pur_m[key]) and not pd.isna(opp_m[key]):
                gap = abs(pur_m[key] - opp_m[key])
                if gap > best_gap and round(gap * 100, 1) > 0:
                    best_gap = gap
                    pur_edge_label = metric_labels[key]
                    pur_edge_pur   = pur_val
                    pur_edge_opp   = opp_val

        if purdue_wins > opp_wins:
            note_color = "#15803d"; note_bg = "#f0fdf4"; note_border = "#22c55e"
            leader = f"✅ Purdue leads {purdue_wins} of {len(metrics)} metrics."
        elif opp_wins > purdue_wins:
            note_color = "#b91c1c"; note_bg = "#fff1f2"; note_border = "#ef4444"
            leader = f"⚠️ Opponent leads {opp_wins} of {len(metrics)} metrics."
        else:
            note_color = "#92400e"; note_bg = "#fffbeb"; note_border = "#f59e0b"
            leader = f"➖ Even matchup — {purdue_wins} of {len(metrics)} metrics each."

        parts = [leader]
        if opp_threat_label:
            parts.append(f"Opponent Advantage: {opp_threat_label} ({opp_threat_opp} vs {opp_threat_pur}) - ")
        if pur_edge_label:
            parts.append(f"Purdue Advantage: {pur_edge_label} ({pur_edge_pur} vs {pur_edge_opp}).")

        scout_note = ui.div(
            " ".join(parts),
            style=(
                f"margin-top:8px; padding:8px 12px; font-size:12px; line-height:1.6;"
                f"color:{note_color}; background:{note_bg};"
                f"border-left:3px solid {note_border}; border-radius:0 4px 4px 0;"
            )
        )

        return ui.div(
            {"class": "team-summary-wrap"},
            ui.div("Team Summary", class_="team-summary-title"),
            ui.tags.table(
                {"class": "team-summary-table"},
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Team"),
                        ui.tags.th("Strike %"),
                        ui.tags.th("Whiff %"),
                        ui.tags.th("Chase %"),
                        ui.tags.th("First Pitch Strike %"),
                    )
                ),
                ui.tags.tbody(
                    ui.tags.tr(*pur_tds),
                    ui.tags.tr(*opp_tds),
                )
            ),
            scout_note,
        )

    
    @output
    @render.ui
    def main_tabs():
        if date_range_invalid():
            return ui.div(
                ui.tags.p(
                    "Start date must be on or before end date.",
                    style="margin: 0; font-weight: 900;",
                ),
                class_="panel",
            )

        return ui.div(
            ui.navset_tab(
                ui.nav_panel(
                    "Home",
                    ui.div(
                        ui.output_ui("home_profile_header"),
                        ui.output_ui("movement_legend"),
                        ui.output_ui("home_content"),
                        class_="panel",
                    ),
                ),
                ui.nav_panel(
                    "Comparison",
                    ui.div(
                        ui.div(
                            ui.input_select("cmp_opponent_team", "Opponent Team", choices={}),
                            ui.output_ui("cmp_player_select"),
                            class_="cmp-filter-row",
                        ),
                        ui.output_ui("cmp_content"),
                        class_="panel",
                    ),
                ),
                ui.nav_panel(
                    "Prediction",
                    ui.div(
                        ui.div("Prediction", class_="profile-title"),
                        ui.div(
                            ui.input_action_button(
                                "retrain_prediction_models",
                                "Retrain Prediction Models",
                                class_="btn btn-sm btn-outline-dark",
                            ),
                            style="display:flex;justify-content:flex-end;margin:2px 0 10px 0;",
                        ),
                        ui.output_ui("prediction_content"),
                        ui.output_ui("prediction_table_section"),
                        class_="panel",
                    ),
                ),

            ),
            class_="tabs-wrap",
        )

    @output
    @render.text
    def player_summary():
        return player_summary_text()

    @output
    @render.ui
    def home_content():
        # HitTrax batter profile (session-aggregated, separate data source)
        if input.data_source() == "hittrax":
            if input.player_type() == "pitcher":
                return ui.div("⚠️  HitTrax only contains batter data. Switch Player Type to Batter to view the profile.", style=(
                    "padding:14px 18px; border-radius:8px; background:#fef9ec;"
                    "border:1.5px solid #DDB945; color:#6b4e00;"
                    "font-size:15px; font-weight:700; margin-top:12px;"
                ))
            d = hittrax_data_active()
            if d is None or d.empty:
                return ui.div(
                    "No HitTrax sessions found for the selected player and date range.",
                    style="text-align:center;margin-top:30px;color:#666;",
                )
            def card_header(title, wrap_id):
                return ui.tags.div(
                    title,
                    ui.tags.span("⛶", class_="expand-btn",
                                 onclick=f"expandChart('{title}','{wrap_id}')",
                                 title="Expand"),
                    style=(
                        "font-size:13px;font-weight:700;color:#444;"
                        "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                        "background:#f3f3f3;border-radius:10px 10px 0 0;"
                        "position:relative;"
                    ),
                )

            return ui.TagList(
                # Season Totals card
                ui.div(
                    ui.div("Season Totals", style=(
                        "font-size:13px;font-weight:700;color:#444;"
                        "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                        "background:#f3f3f3;border-radius:10px 10px 0 0;"
                    )),
                    ui.output_ui("hittrax_aggregate_stats"),
                    class_="card",
                    style="margin-bottom:14px;",
                ),
                # Max Distance per Session (full width, click-to-select)
                ui.div(
                    card_header("Max Distance per Session", "hittrax-dist-wrap"),
                    ui.tags.div(
                        ui.output_plot("hittrax_max_distance_plot", height="380px", click=True),
                        id="hittrax-dist-wrap",
                    ),
                    class_="card",
                    style="margin-bottom:14px;",
                ),
                # Batted Ball Mix (full width, click-to-select)
                ui.div(
                    card_header("Batted Ball Mix (per session)", "hittrax-mix-wrap"),
                    ui.tags.div(
                        ui.output_plot("hittrax_batted_ball_mix_plot", height="380px", click=True),
                        id="hittrax-mix-wrap",
                    ),
                    class_="card",
                    style="margin-bottom:14px;",
                ),
                # Session Log (full width)
                ui.div(
                    ui.div("Session Log", style=(
                        "font-size:13px;font-weight:700;color:#444;"
                        "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                        "background:#f3f3f3;border-radius:10px 10px 0 0;"
                    )),
                    ui.output_ui("hittrax_session_log"),
                    class_="card",
                ),
            )

        df = current_df()
        team = input.team()
        if df is None or df.empty:
            return ui.div("No data found for the selected season and date range.")
        if not team:
            return ui.div("Select a team to view data.")

        warning = session_player_type_warning()
        if warning:
            return ui.div(warning, style=(
                "padding:14px 18px; border-radius:8px; background:#fef9ec;"
                "border:1.5px solid #DDB945; color:#6b4e00;"
                "font-size:15px; font-weight:700; margin-top:12px;"
            ))

        if input.player_type() == "batter":
            data = batter_data()
            if data is None or data.empty:
                return ui.div(
                    "No batter data for the selected filters.",
                    style="text-align:center;margin-top:30px;color:#666;",
                )
            session  = input.session_type()
            is_bp    = (session == "batting_practice")
            bot_plot = "batter_ev_dist_plot" if is_bp else "batter_radar_plot"
            bot_title = "Exit Velo Distribution" if is_bp else "Plate Discipline"
            PLOT_H = "450px"
            SPRAY_H = "260px"
            LOC_H = "670px"

            return ui.TagList(
                ui.div(
                    # Batting Summary (total row)
                    ui.div(
                        ui.div(
                            ui.tags.span("Batting Summary", style="font-size:13px;font-weight:700;color:#444;"),
                            ui.tags.span(
                                ui.tags.span(style="display:inline-block;width:10px;height:10px;border-radius:2px;background:rgba(39,174,96,0.18);border:1px solid #1a7a3a;margin-right:3px;vertical-align:middle;"),
                                ui.tags.span("Top third", style="font-size:10px;color:#888;margin-right:12px;vertical-align:middle;"),
                                ui.tags.span(style="display:inline-block;width:10px;height:10px;border-radius:2px;background:rgba(241,196,15,0.22);border:1px solid #8a6d00;margin-right:3px;vertical-align:middle;"),
                                ui.tags.span("Middle third", style="font-size:10px;color:#888;margin-right:12px;vertical-align:middle;"),
                                ui.tags.span(style="display:inline-block;width:10px;height:10px;border-radius:2px;background:rgba(231,76,60,0.15);border:1px solid #c0392b;margin-right:3px;vertical-align:middle;"),
                                ui.tags.span("Bottom third", style="font-size:10px;color:#888;vertical-align:middle;"),
                                style="float:right;",
                            ),
                            style=(
                                "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                                "background:#f3f3f3;border-radius:10px 10px 0 0;"
                                "display:flex;justify-content:space-between;align-items:center;"
                            ),
                        ),
                        ui.div(ui.output_table("batter_batting_line"), class_="usage-table-wrap"),
                        class_="card",
                        style="margin-bottom:14px;",
                    ),
                    # Summary Table (per-pitch breakdown, clickable)
                    ui.div(
                        ui.div("Summary Table", style=(
                            "font-size:13px;font-weight:700;color:#444;"
                            "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                            "background:#f3f3f3;border-radius:10px 10px 0 0;"
                        )),
                        ui.div(ui.output_table("batter_pitch_table"), class_="usage-table-wrap"),
                        ui.tags.script(ui.HTML(
                            "$(document).on('shiny:value', function(e) {"
                            "  if (e.name !== 'batter_pitch_table') return;"
                            "  setTimeout(function() {"
                            "    $('#batter_pitch_table tbody tr').css('cursor','pointer').off('click').on('click', function() {"
                            "      var pt = $(this).find('td:first').text().trim();"
                            "      if (pt) Shiny.setInputValue('clicked_pitch', pt, {priority:'event'});"
                            "    });"
                            "  }, 100);"
                            "});"
                        )),
                        class_="card",
                        style="margin-bottom:14px;",
                    ),
                    # Row 3: Out Rate + Pitch Results (side by side, full width)
                    ui.div(
                        # Out Rate by Zone
                        ui.div(
                            ui.tags.div(
                                ui.tags.span(
                                    "Out Rate by Zone",
                                    ui.tags.span("i", class_="tip-icon"),
                                    class_="stat-tip",
                                    **{"data-stat-key": "__out_rate__"},
                                ),
                                ui.tags.span("⛶", class_="expand-btn",
                                             onclick="expandChart('Out Rate by Zone','batter-outrate-wrap')",
                                             title="Expand"),
                                style=(
                                    "font-size:13px;font-weight:700;color:#444;"
                                    "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                                    "background:#f3f3f3;border-radius:10px 10px 0 0;"
                                    "position:relative;"
                                ),
                            ),
                            ui.tags.div(
                                ui.output_plot("batter_out_rate_plot", height="600px"),
                                id="batter-outrate-wrap",
                            ),
                            ui.div(
                                ui.tags.div(
                                    ui.tags.span("Red", style="color:#c0392b;font-weight:700;"),
                                    ": danger zone, gets out 75%+ of the time",
                                ),
                                ui.tags.div(
                                    ui.tags.span("Gold", style="color:#d4a017;font-weight:700;"),
                                    ": gets out more often than not (50 - 75%)",
                                ),
                                ui.tags.div(
                                    ui.tags.span("Light", style="color:#b8860b;font-weight:700;"),
                                    ": moderate (25 - 50%)",
                                ),
                                ui.tags.div(
                                    ui.tags.span("White", style="color:#888;font-weight:700;"),
                                    ": productive zone (< 25%)",
                                ),
                                style=(
                                    "font-size:11px;color:#666;line-height:1.5;"
                                    "padding:10px 14px;border-top:1px solid #e0e0e0;"
                                    "background:#fafafa;"
                                ),
                            ),
                            class_="card",
                        ),
                        # Pitch Results scatter
                        ui.div(
                            ui.tags.div(
                                "Pitch Results",
                                ui.tags.span("⛶", class_="expand-btn",
                                             onclick="expandChart('Pitch Results','batter-pitchresults-wrap')",
                                             title="Expand"),
                                style=(
                                    "font-size:13px;font-weight:700;color:#444;"
                                    "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                                    "background:#f3f3f3;border-radius:10px 10px 0 0;"
                                    "position:relative;"
                                ),
                            ),
                            ui.tags.div(
                                ui.output_ui("batter_pitch_results_plot"),
                                id="batter-pitchresults-wrap",
                            ),
                            class_="card",
                        ),
                        style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;",
                    ),
                    # Row 4: Spray Chart | Exit Velo vs LA | Plate Discipline (3 columns)
                    ui.div(
                        ui.div(
                            ui.tags.div(
                                "Spray Chart",
                                ui.tags.span("⛶", class_="expand-btn",
                                             onclick="expandChart('Spray Chart','batter-spray-wrap')",
                                             title="Expand"),
                                style=(
                                    "font-size:13px;font-weight:700;color:#444;"
                                    "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                                    "background:#f3f3f3;border-radius:10px 10px 0 0;"
                                    "position:relative;"
                                ),
                            ),
                            ui.tags.div(
                                ui.output_plot("batter_spray_plot", height=PLOT_H),
                                id="batter-spray-wrap",
                            ),
                            class_="card",
                        ),
                        ui.div(
                            ui.tags.div(
                                "Exit Velo vs. Launch Angle",
                                ui.tags.span("⛶", class_="expand-btn",
                                             onclick="expandChart('Exit Velo vs. Launch Angle','batter-evla-wrap')",
                                             title="Expand"),
                                style=(
                                    "font-size:13px;font-weight:700;color:#444;"
                                    "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                                    "background:#f3f3f3;border-radius:10px 10px 0 0;"
                                    "position:relative;"
                                ),
                            ),
                            ui.tags.div(
                                ui.output_plot("batter_ev_la_plot", height=PLOT_H),
                                id="batter-evla-wrap",
                            ),
                            class_="card",
                        ),
                        ui.div(
                            ui.tags.div(
                                bot_title,
                                ui.tags.span("⛶", class_="expand-btn",
                                             onclick=f"expandChart('{bot_title}','batter-bot-wrap')",
                                             title="Expand"),
                                style=(
                                    "font-size:13px;font-weight:700;color:#444;"
                                    "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                                    "background:#f3f3f3;border-radius:10px 10px 0 0;"
                                    "position:relative;"
                                ),
                            ),
                            ui.tags.div(
                                ui.output_plot(bot_plot, height=PLOT_H),
                                id="batter-bot-wrap",
                            ),
                            class_="card",
                        ),
                        style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px;",
                    ),
                    # Scout Insights (full-width)
                    ui.div(
                        ui.div("Scout Insights", style=(
                            "font-size:13px;font-weight:700;color:#444;"
                            "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                            "background:#f3f3f3;border-radius:10px 10px 0 0;"
                        )),
                        ui.output_ui("batter_scout_insights"),
                        class_="card",
                    ),
                ),
            )



        # Pitcher profile
        data = pitcher_data()
        if data is None or data.empty:
            return ui.div("No pitcher data for the selected filters.")

        session = input.session_type()
        is_bullpen = (session == "bullpen")

        rows = [
            ui.row(
                ui.column(4, ui.card(
                    ui.tags.div(
                        "Pitch Usage",
                        ui.tags.span("⛶", class_="expand-btn",
                                     onclick="expandChart('Pitch Usage','pie')",
                                     title="Expand"),
                        class_="card-header", style="position:relative;",
                    ),
                    ui.tags.div(ui.output_plot("pie", height="340px"), id="pie"),
                )),
                ui.column(4, ui.card(
                    ui.tags.div(
                        "Pitch Locations",
                        ui.tags.span("⛶", class_="expand-btn",
                                     onclick="expandChart('Pitch Locations','location-wrap')",
                                     title="Expand"),
                        class_="card-header", style="position:relative;",
                    ),
                    ui.tags.div(ui.output_ui("location"), id="location-wrap"),
                )),
                ui.column(4, ui.card(
                    ui.tags.div(
                        "Pitch Movements",
                        ui.tags.span("⛶", class_="expand-btn",
                                     onclick="expandChart('Pitch Movements','movement-wrap')",
                                     title="Expand"),
                        class_="card-header", style="position:relative;",
                    ),
                    ui.tags.div(ui.output_ui("movement"), id="movement-wrap"),
                )),
            ),
            ui.row(
                ui.column(12, ui.card(
                    ui.card_header("Summary Table"),
                    ui.div(ui.output_table("usage_table"), class_="usage-table-wrap"),
                )),
            ),
        ]

        if not is_bullpen:
            rows.append(ui.row(
                ui.column(12, ui.card(
                    ui.card_header("Pitcher Performance by Batter Approach"),
                    ui.output_ui("pitcher_approach_table"),
                )),
            ))
            rows.append(ui.row(
                ui.column(12, ui.card(
                    ui.card_header("Scout Insights"),
                    ui.output_ui("pitcher_scout_insights"),
                )),
            ))

        return ui.div(*rows)

    # -----------------------------------------------------------------------
    # Plots + table
    # -----------------------------------------------------------------------
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
            plt.close(fig)
            return fig

        labels = usage[PITCH_TYPE_COL].tolist()
        pcts = usage["usage_pct"].values
        sel = selected_pitch.get()
        wedge_colors = [colors.get(pt, "#888888") for pt in labels]

        wedges, texts, autotexts = ax.pie(
            pcts,
            labels=None,
            colors=wedge_colors,
            startangle=90,
            counterclock=False,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
            pctdistance=0.65,
            textprops={"fontsize": 10, "fontweight": "bold"},
        )
        
        for wedge, pt in zip(wedges, labels):
            wedge.set_alpha(pitch_alpha(pt, sel))

        for t in ax.texts:
            t.set_fontsize(8)

        plt.close(fig)    
        return fig

    @output
    @render.ui
    def location():
        loc = pitcher_loc_data()
        colors = pitch_colors()

        fig = go.Figure()

        if loc is None or loc.empty:
            fig.add_annotation(
                text="No pitch location data",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="#555555"),
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(
                paper_bgcolor="#f7f7f7", plot_bgcolor="#f7f7f7",
                margin=dict(l=20, r=20, t=20, b=20), height=340,
            )
            return ui.HTML(fig.to_html(
                full_html=False, include_plotlyjs=False,
                config={"displayModeBar": False},
            ))

        sel = selected_pitch.get()

        # Shadow zone
        fig.add_shape(type="rect",
            x0=ZONE_LEFT - 0.3, y0=ZONE_BOTTOM - 0.3,
            x1=ZONE_RIGHT + 0.3, y1=ZONE_TOP + 0.3,
            fillcolor="rgba(200,200,200,0.25)", line=dict(width=0),
            layer="below",
        )
        # Strike zone
        fig.add_shape(type="rect",
            x0=ZONE_LEFT, y0=ZONE_BOTTOM,
            x1=ZONE_RIGHT, y1=ZONE_TOP,
            line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)",
        )
        # Cross lines
        mid_y = (ZONE_BOTTOM + ZONE_TOP) / 2
        fig.add_shape(type="line",
            x0=ZONE_LEFT, y0=mid_y, x1=ZONE_RIGHT, y1=mid_y,
            line=dict(color="gray", width=1, dash="dash"),
        )
        fig.add_shape(type="line",
            x0=0, y0=ZONE_BOTTOM, x1=0, y1=ZONE_TOP,
            line=dict(color="gray", width=1, dash="dash"),
        )

        # Home plate
        hp_x = [-0.35, -0.35, 0, 0.35, 0.35]
        hp_y = [0.3, 0.1, 0, 0.1, 0.3]
        fig.add_trace(go.Scatter(
            x=hp_x + [hp_x[0]], y=hp_y + [hp_y[0]],
            mode="lines", fill="toself",
            fillcolor="rgba(255,255,255,0.8)",
            line=dict(color="#555", width=1),
            showlegend=False, hoverinfo="skip",
        ))

        # Pitch dots
        for pt, g in loc.groupby(PITCH_TYPE_COL):
            color = colors.get(pt, "#888888")
            is_selected = (sel == pt)
            opacity = 1.0 if (not sel or is_selected) else 0.15
            size = 10 if is_selected else 7

            g_plot = g.sample(n=min(len(g), 50), random_state=42) if len(g) > 50 else g

            fig.add_trace(go.Scatter(
                x=g_plot["PlateLocSide"],
                y=g_plot["PlateLocHeight"],
                mode="markers",
                name=pt,
                showlegend=False,
                hovertemplate=(
                    f"{pt}<br>"
                    "Side: %{x:.2f}<br>"
                    "Height: %{y:.2f}<extra></extra>"
                ),
                marker=dict(
                    size=size, color=color, opacity=opacity,
                    line=dict(
                        color="black" if (not sel or is_selected) else color,
                        width=0.5 if (not sel or is_selected) else 0,
                    ),
                ),
            ))

        fig.update_xaxes(
            range=[-3, 3], showgrid=True,
            gridcolor="rgba(0,0,0,0.08)", zeroline=False,
            title="PlateLocSide",
        )
        fig.update_yaxes(
            range=[0, 5], showgrid=True,
            gridcolor="rgba(0,0,0,0.08)", zeroline=False,
            scaleanchor="x", scaleratio=1,
            title="PlateLocHeight",
        )
        fig.update_layout(
            paper_bgcolor="#f7f7f7", plot_bgcolor="#f7f7f7",
            margin=dict(l=10, r=10, t=10, b=10),
            height=340, dragmode="zoom",
        )

        return ui.HTML(fig.to_html(
            full_html=False, include_plotlyjs=False,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": [
                    "toImage", "select2d", "lasso2d",
                    "hoverClosestCartesian", "hoverCompareCartesian",
                    "toggleSpikelines",
                ],
            },
        ))


    @output
    @render.ui
    def movement_legend():
        order = pitch_order()
        if not order:
            return ui.div()
        cols = pitch_colors()
        sel = selected_pitch.get()

        items = []
        for pt in order:
            if pt.lower() in {"other", "undefined"}:
                continue

            color = cols.get(pt, "#777777")
            is_active = (sel == "" or sel == pt)

            items.append(
                ui.tags.span(
                    ui.span(style=f"display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};"),
                    ui.span(pt),
                    style=(
                        f"display:inline-flex;align-items:center;gap:6px;cursor:pointer;"
                        f"opacity:{'1.0' if is_active else '0.25'};"
                        f"font-weight:{'900' if sel == pt else '400'};"
                        f"padding:3px 8px;border-radius:6px;"
                        f"{'border:1.5px solid #DDB945;background:#fffbef;' if sel == pt else 'border:1.5px solid transparent;'}"
                    ),
                    onclick=f"Shiny.setInputValue('clicked_pitch', '{pt}', {{priority: 'event'}})",
                )
            )

        reset_btn = ui.tags.button(
            "Reset",
            type="button",
            onclick="Shiny.setInputValue('reset_pitch', Date.now(), {priority: 'event'})",
            style=(
                "margin-left:12px;padding:4px 10px;border:1px solid #cfcfcf;"
                "border-radius:6px;background:#ffffff;font-weight:700;cursor:pointer;"
            ),
        )
        return ui.div(*items, reset_btn, class_="legend-row")


    @output
    @render.table
    def usage_table():
        data = pitcher_data()
        pid = input.player() if input.player_type() == "pitcher" else None
        session_type = input.session_type()
        sel = selected_pitch.get()

        bullpen_cols = [
            "Pitch type", "Count", "Usage %",
            "Max Velo", "Avg Velo", "Spin Rate", "IVB Avg", "HB Avg", "Strike %"
        ]

        live_scrimmage_cols = [
            "Pitch type", "Count", "Usage %",
            "Max Velo", "Avg Velo", "Spin Rate",
            "IVB Avg", "HB Avg",
            "Strike %", "Called Strike %",
            "Swing %", "SwStrike %",
            "Whiff %", "Zone Swing %",
            "Zone Contact %", "Chase %",
            "Chase Contact %"
        ]

        cols = bullpen_cols if session_type == "bullpen" else live_scrimmage_cols

        if data is None or data.empty or not pid:
            return pd.DataFrame(columns=cols)

        summary = cached_pitch_metrics()
        if summary is None or summary.empty:
            return pd.DataFrame(columns=cols)

        out = summary.copy()

        out["max_velo"] = pd.to_numeric(out.get("max_velo"), errors="coerce").round(1)
        out["avg_velo"] = pd.to_numeric(out.get("avg_velo"), errors="coerce").round(1)
        out["spin_rate"] = pd.to_numeric(out.get("spin_rate"), errors="coerce").round(0)
        out["ivb_avg"] = out["ivb_avg"].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
        out["hb_avg"]  = out["hb_avg"].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")

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
            "ivb_avg": "IVB Avg",
            "hb_avg": "HB Avg",
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

        out = out[cols]

        def highlight_col(col):
            is_sel_row = (
                out["Pitch type"].astype(str).str.strip() == str(sel).strip()
                if sel else pd.Series(False, index=out.index)
            )

            # Selected row style — warm yellow bg + gold left border on first col
            # !important is required because .usage-table-wrap CSS uses !important
            # on zebra-stripe background and would otherwise override inline styles.
            selected_style_main = (
                "font-size:18px !important;font-weight:800 !important;color:#000000 !important;"
                "background-color:#fff8e1 !important;"
                "border-left:4px solid #DDB945 !important;"
                "padding:12px 10px !important;"
            )
            selected_style_other = (
                "font-size:18px !important;font-weight:800 !important;color:#000000 !important;"
                "background-color:#fff8e1 !important;"
                "padding:12px 10px !important;"
            )

            # Normal row style
            normal_style_main = "font-size:12.5px;color:#444444;"
            normal_style_other = "font-size:12px;color:#555555;"

            if col.name == "Pitch type":
                return [selected_style_main if v else normal_style_main for v in is_sel_row]

            return [selected_style_other if v else normal_style_other for v in is_sel_row]

        return (
            out.style
            .hide(axis="index")
            .format({
                "Max Velo": "{:.1f}",
                "Avg Velo": "{:.1f}",
                "Spin Rate": "{:.0f}",
            })
            .apply(highlight_col, axis=0)
        )


    # ── pitcher performance by batter approach ─────────────────────────────
    @output
    @render.ui
    def pitcher_approach_table():
        data = pitcher_data()
        pid = input.player() if input.player_type() == "pitcher" else None
        session_type = input.session_type()

        # Only show for sessions with real batters
        if session_type in ("bullpen", "batting_practice"):
            return ui.div()
        if data is None or data.empty or not pid:
            return ui.div()

        median_sw = batter_swing_rate_median()
        if median_sw is None:
            return ui.div()

        # Classify each batter as aggressive or passive
        SW = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        TERMINAL = {"Single", "Double", "Triple", "HomeRun", "Out", "FieldersChoice",
                    "Error", "Walk", "Strikeout", "HitByPitch", "SacrificeFly",
                    "SacrificeBunt", "Sacrifice", "CatcherInterference"}
        PITCHER_WIN = {"Out", "Strikeout", "FieldersChoice", "SacrificeFly",
                       "SacrificeBunt", "Sacrifice", "Error"}
        BIP_RESULTS = {"Single", "Double", "Triple", "HomeRun", "Out",
                       "FieldersChoice", "Error", "Sacrifice"}
        BIP_OUTS = {"Out", "FieldersChoice", "Error", "Sacrifice"}

        batter_ids = data["BatterId"].dropna().unique()
        aggressive_bids = set()
        passive_bids = set()

        df_all = current_df()
        if df_all is None or df_all.empty:
            return ui.div()

        for bid in batter_ids:
            bd = df_all[df_all["BatterId"] == bid]
            n = len(bd)
            if n < 10:
                continue
            pc = bd["PitchCall"].astype(str).str.strip()
            sw_rate = pc.isin(SW).sum() / n
            if sw_rate > median_sw:
                aggressive_bids.add(bid)
            else:
                passive_bids.add(bid)

        def compute_group(d):
            if d.empty:
                return {"Count": 0, "Win": 0, "Win%": "—", "Ks": 0, "BBs": 0,
                        "BIP": 0, "Out Rate": "—", "BIP Outs": 0}
            pr = d["PlayResult"].astype(str).str.strip() if "PlayResult" in d.columns else pd.Series("", index=d.index)
            pc = d["PitchCall"].astype(str).str.strip() if "PitchCall" in d.columns else pd.Series("", index=d.index)
            korbb = d["KorBB"].astype(str).str.strip() if "KorBB" in d.columns else pd.Series("", index=d.index)

            # Terminal pitches: PlayResult has a result, OR KorBB has K/BB, OR HBP
            is_terminal = (
                pr.isin(TERMINAL) |
                korbb.isin({"Strikeout", "Walk"}) |
                pc.eq("HitByPitch")
            )
            t = d[is_terminal]
            if t.empty:
                return {"Count": 0, "Win": 0, "Win%": "—", "Ks": 0, "BBs": 0,
                        "BIP": 0, "Out Rate": "—", "BIP Outs": 0}

            t_pr = t["PlayResult"].astype(str).str.strip()
            t_korbb = t["KorBB"].astype(str).str.strip() if "KorBB" in t.columns else pd.Series("", index=t.index)

            # Use KorBB for Ks and BBs (Trackman stores them there)
            ks = int(t_korbb.eq("Strikeout").sum())
            bbs = int(t_korbb.eq("Walk").sum())

            # Count unique PAs (use PitchofPA == 1 if available)
            if "PitchofPA" in d.columns:
                count = int((pd.to_numeric(d["PitchofPA"], errors="coerce") == 1).sum())
            else:
                count = len(t)
            if count == 0:
                count = len(t)

            # Win = pitcher got the batter out (K, out, FC, sac, error)
            win = ks + int(t_pr.isin({"Out", "FieldersChoice", "SacrificeFly", "SacrificeBunt", "Error"}).sum())

            # BIP = balls put in play
            bip = int(t_pr.isin(BIP_RESULTS).sum())
            bip_outs = int(t_pr.isin(BIP_OUTS).sum())

            win_pct = f"{win / count * 100:.1f}%" if count > 0 else "—"
            out_rate = f"{bip_outs / bip * 100:.1f}%" if bip > 0 else "—"

            return {"Count": count, "Win": win, "Win%": win_pct, "Ks": ks, "BBs": bbs,
                    "BIP": bip, "Out Rate": out_rate, "BIP Outs": bip_outs}

        passive_data = data[data["BatterId"].isin(passive_bids)]
        aggressive_data = data[data["BatterId"].isin(aggressive_bids)]

        p = compute_group(passive_data)
        a = compute_group(aggressive_data)

        if p["Count"] == 0 and a["Count"] == 0:
            return ui.div()

        # Build two side-by-side cards
        th_dark = (
            "padding:8px 12px;text-align:center;font-weight:700;font-size:11px;"
            "background:#1a1a1a;color:#fff;"
        )
        td_style = "padding:8px 12px;text-align:center;font-size:13px;border-bottom:1px solid #eee;color:#333;"
        td_bold = td_style + "font-weight:800;"
        card_header_style = (
            "font-size:13px;font-weight:700;color:#444;"
            "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
            "background:#f3f3f3;border-radius:10px 10px 0 0;"
            "display:flex;align-items:center;"
        )

        cols = ["Count", "Win", "Win%", "Ks", "BBs", "BIP", "Out Rate", "BIP Outs"]
        bold_cols = {"Count", "Win%"}

        def _build_card(title, grp, tip_key):
            cells = []
            for c in cols:
                s = td_bold if c in bold_cols else td_style
                cells.append(ui.tags.td(str(grp[c]), style=s))

            tbl = ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(*[ui.tags.th(c, style=th_dark) for c in cols]),
                ),
                ui.tags.tbody(ui.tags.tr(*cells)),
                style="width:100%;border-collapse:collapse;",
            )

            header = ui.tags.div(
                title,
                ui.tags.span("i", class_="tip-icon", style="margin-left:6px;"),
                style=card_header_style,
                class_="stat-tip",
                **{"data-stat-key": tip_key},
            )

            return ui.tags.div(
                header, tbl,
                style=(
                    "background:#fff;border:1px solid #e0e0e0;"
                    "border-radius:10px;overflow:hidden;flex:1;"
                ),
            )

        return ui.div(
            _build_card("Passive Hitters", p, "__passive__"),
            _build_card("Aggressive Hitters", a, "__aggressive__"),
            style="display:flex;gap:14px;",
        )

    # ── pitcher scout insights (outcome + count-based) ───────────────────────
    @output
    @render.ui
    def pitcher_scout_insights():
        data = pitcher_data()
        pid = input.player() if input.player_type() == "pitcher" else None
        session_type = input.session_type()
        if data is None or data.empty or not pid:
            return ui.div()
        if session_type == "bullpen":
            return ui.div()

        d = data.copy()
        if PITCH_TYPE_COL not in d.columns or "PitchCall" not in d.columns:
            return ui.div()

        d = d[is_valid_pitch_type(d[PITCH_TYPE_COL])].copy()
        if len(d) < 20:
            return ui.div(
                ui.tags.span("Insufficient data for scouting insights.",
                             style="font-size:12px;color:#888;font-style:italic;"),
                style="padding:12px 16px;",
            )

        pc = d["PitchCall"].astype(str).str.strip()
        pr = d["PlayResult"].astype(str).str.strip() if "PlayResult" in d.columns else pd.Series("", index=d.index)

        SW = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        STRIKE_EVENTS = {"StrikeCalled", "StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        OUT_RESULTS = {"Out", "FieldersChoice", "Strikeout", "Sacrifice", "SacrificeFly", "SacrificeBunt", "Error"}
        PA_RESULTS = {"Single", "Double", "Triple", "HomeRun", "Out", "FieldersChoice",
                      "Strikeout", "Walk", "HitByPitch", "SacrificeFly", "SacrificeBunt",
                      "Sacrifice", "Error", "CatcherInterference"}
        HIT_RESULTS = {"Single", "Double", "Triple", "HomeRun"}

        is_terminal = pr.isin(PA_RESULTS) | pc.eq("HitByPitch")

        def _badge(badge, color, desc):
            return ui.tags.div(
                ui.tags.span(badge, style=(
                    f"display:inline-block;background:{color};color:#fff;"
                    "font-size:9px;font-weight:800;letter-spacing:0.5px;"
                    "padding:2px 8px;border-radius:3px;margin-right:10px;"
                    "white-space:nowrap;"
                )),
                ui.tags.span(desc, style="font-size:12px;line-height:1.5;color:#444;"),
                style="display:flex;align-items:center;flex-wrap:wrap;padding:6px 0;border-bottom:1px solid #f0f0f0;",
            )

        def _section(title, items):
            if not items:
                return None
            return ui.tags.div(
                ui.tags.div(title, style=(
                    "font-size:11px;font-weight:700;color:#888;text-transform:uppercase;"
                    "letter-spacing:0.5px;padding-bottom:4px;margin-bottom:4px;"
                    "border-bottom:1px solid #eee;"
                )),
                *items,
                style="margin-bottom:24px;padding-bottom:12px;border-bottom:2px solid #e8e8e8;",
            )

        sections = []

        # ── 1. Count-Based Recommendations ──
        count_items = []
        if "Balls" in d.columns and "Strikes" in d.columns:
            balls = pd.to_numeric(d["Balls"], errors="coerce")
            strikes = pd.to_numeric(d["Strikes"], errors="coerce")

            COUNT_GROUPS = [
                ("First Pitch (0-0)", [(0, 0)], "strike_rate"),
                ("Ahead (0-2, 1-2)", [(0, 2), (1, 2)], "whiff_rate"),
                ("Behind (2-0, 3-0, 3-1)", [(2, 0), (3, 0), (3, 1)], "strike_rate"),
                ("Even (1-1, 2-2)", [(1, 1), (2, 2)], "out_rate"),
            ]

            for label, counts, metric in COUNT_GROUPS:
                count_mask = pd.Series(False, index=d.index)
                for b, s in counts:
                    count_mask = count_mask | (balls.eq(b) & strikes.eq(s))

                count_d = d[count_mask]
                if len(count_d) < 10:
                    continue

                count_pc = count_d["PitchCall"].astype(str).str.strip()
                count_pr = count_d["PlayResult"].astype(str).str.strip() if "PlayResult" in count_d.columns else pd.Series("", index=count_d.index)
                count_terminal = count_pr.isin(PA_RESULTS) | count_pc.eq("HitByPitch")

                best_pt = None
                best_val = -1
                best_n = 0

                for pt, g in count_d.groupby(PITCH_TYPE_COL):
                    if len(g) < 3:
                        continue
                    g_pc = g["PitchCall"].astype(str).str.strip()
                    g_pr = g["PlayResult"].astype(str).str.strip() if "PlayResult" in g.columns else pd.Series("", index=g.index)
                    g_terminal = g_pr.isin(PA_RESULTS) | g_pc.eq("HitByPitch")

                    if metric == "strike_rate":
                        val = g_pc.isin(STRIKE_EVENTS).mean()
                        val_label = "strike rate"
                    elif metric == "whiff_rate":
                        n_sw = g_pc.isin(SW).sum()
                        val = (g_pc.eq("StrikeSwinging").sum() / n_sw) if n_sw > 0 else 0
                        val_label = "whiff rate"
                    else:  # out_rate
                        n_pa = int(g_terminal.sum())
                        n_outs = int((g_terminal & g_pr.isin(OUT_RESULTS)).sum())
                        val = n_outs / n_pa if n_pa > 0 else 0
                        val_label = "out rate"

                    if val > best_val:
                        best_val = val
                        best_pt = pt
                        best_n = len(g)

                if best_pt and best_val > 0:
                    count_items.append(_badge(label.split("(")[0].strip().upper(), "#185FA5",
                        f"{best_pt} — {best_val:.0%} {val_label} ({best_n} pitches). "
                        f"Best option in {label.split('(')[1].rstrip(')').strip()} counts."))

        sec = _section("By Count", count_items[:4])
        if sec:
            sections.append(sec)

        # ── 3. Tendencies (not visible in Summary Table) ──
        tend_items = []

        # First-pitch tendency
        if "Balls" in d.columns and "Strikes" in d.columns:
            b_col = pd.to_numeric(d["Balls"], errors="coerce")
            s_col = pd.to_numeric(d["Strikes"], errors="coerce")
            fp = d[b_col.eq(0) & s_col.eq(0)]
            if len(fp) >= 10:
                fp_pc = fp["PitchCall"].astype(str).str.strip()
                fp_strike_pct = fp_pc.isin(STRIKE_EVENTS).mean()
                # Most common first pitch type
                fp_valid = fp[is_valid_pitch_type(fp[PITCH_TYPE_COL])]
                if not fp_valid.empty:
                    fp_top = fp_valid[PITCH_TYPE_COL].value_counts()
                    fp_primary = fp_top.index[0]
                    fp_primary_pct = fp_top.iloc[0] / len(fp_valid)
                    if fp_strike_pct >= 0.65:
                        tend_items.append(_badge("FIRST PITCH", "#1D9E75",
                            f"Throws {fp_primary} first {fp_primary_pct:.0%} of the time with "
                            f"{fp_strike_pct:.0%} first-pitch strike rate. Aggressive — attacks early."))
                    elif fp_strike_pct <= 0.45:
                        tend_items.append(_badge("FIRST PITCH", "#E67E22",
                            f"Only {fp_strike_pct:.0%} first-pitch strike rate. "
                            f"Favors {fp_primary} ({fp_primary_pct:.0%}) — tends to fall behind."))
                    else:
                        tend_items.append(_badge("FIRST PITCH", "#185FA5",
                            f"Leads with {fp_primary} ({fp_primary_pct:.0%}), "
                            f"{fp_strike_pct:.0%} first-pitch strike rate."))

        # Command pattern (zone rate + arm-side/glove-side)
        if "PlateLocSide" in d.columns and "PlateLocHeight" in d.columns:
            ls = pd.to_numeric(d["PlateLocSide"], errors="coerce")
            lh = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
            valid_loc = ls.notna() & lh.notna()
            dv = d[valid_loc]

            if len(dv) >= 20:
                in_zone = (
                    ls[valid_loc].between(ZONE_LEFT, ZONE_RIGHT) &
                    lh[valid_loc].between(ZONE_BOTTOM, ZONE_TOP)
                )
                zone_pct = in_zone.mean()

                if zone_pct >= 0.55:
                    tend_items.append(_badge("ZONE CONTROL", "#1D9E75",
                        f"{zone_pct:.0%} of pitches land in the zone. "
                        "Attacks hitters aggressively — forces swings."))
                elif zone_pct <= 0.35:
                    tend_items.append(_badge("NIBBLER", "#2980B9",
                        f"Only {zone_pct:.0%} in the zone. "
                        "Works the edges — patient hitters can wait her out."))

                # Arm-side vs glove-side
                arm_side = (ls[valid_loc] > 0).mean()
                glove_side = (ls[valid_loc] < 0).mean()
                if arm_side >= 0.62:
                    tend_items.append(_badge("ARM-SIDE HEAVY", "#E67E22",
                        f"{arm_side:.0%} of pitches land arm-side. "
                        "Predictable location — batters can cheat inside."))
                elif glove_side >= 0.62:
                    tend_items.append(_badge("GLOVE-SIDE HEAVY", "#E67E22",
                        f"{glove_side:.0%} of pitches land glove-side. "
                        "Favors the away side — look for pitches off the plate."))

        # Pitch sequencing — does she repeat pitches or mix?
        if PITCH_TYPE_COL in d.columns and len(d) >= 30:
            pts = d[is_valid_pitch_type(d[PITCH_TYPE_COL])][PITCH_TYPE_COL].astype(str).str.strip()
            if len(pts) >= 20:
                repeats = sum(1 for a, b in zip(pts.iloc[:-1], pts.iloc[1:]) if a == b)
                repeat_rate = repeats / (len(pts) - 1)
                n_types = pts.nunique()

                if repeat_rate >= 0.45:
                    tend_items.append(_badge("REPEATER", "#854F0B",
                        f"Throws the same pitch back-to-back {repeat_rate:.0%} of the time. "
                        "Predictable sequencing — sit on the pitch she just threw."))
                elif repeat_rate <= 0.20 and n_types >= 4:
                    tend_items.append(_badge("MIXER", "#1D9E75",
                        f"Only {repeat_rate:.0%} back-to-back repeats across {n_types} pitch types. "
                        "Keeps hitters off-balance with varied sequencing."))

        sec = _section("Tendencies", tend_items[:4])
        if sec:
            sections.append(sec)

        if not sections:
            return ui.div(
                ui.tags.span("Insufficient data for scouting insights.",
                             style="font-size:12px;color:#888;font-style:italic;"),
                style="padding:12px 16px;",
            )

        return ui.div(*sections, style="padding:12px 16px;")

    # ── dynamic home tab header ──────────────────────────────────────────────
    @output
    @render.ui
    def home_profile_header():
        if input.data_source() == "hittrax":
            return ui.div(
                ui.div("Batter Profile", class_="profile-title"),
                ui.div(hittrax_summary_text() or "Select a batter to view profile", class_="player-summary"),
            )
        if input.player_type() == "pitcher":
            return ui.div(
                ui.div("Pitcher Profile", class_="profile-title"),
                ui.div(ui.output_text("player_summary"), class_="player-summary"),
            )
        txt = batter_summary_text()
        return ui.div(
            ui.div("Batter Profile", class_="profile-title"),
            ui.div(txt or "Select a batter to view profile", class_="player-summary"),
        )

    # ── batting summary (total row, same style as pitcher table) ────────────
    @output
    @render.table
    def batter_batting_line():
        data = batter_data()
        bid  = input.player() if input.player_type() == "batter" else None
        if data is None or data.empty or not bid:
            return pd.DataFrame()

        s       = compute_batter_stats(data, bid)
        session = input.session_type()
        is_bp   = (session == "batting_practice")

        def fmt(v):
            if v is None:
                return "—"
            if v >= 1.0:
                return f"{v:.3f}"
            return f".{int(round(v * 1000)):03d}"

        if is_bp:
            row = {"": "Total", "AB": s["AB"], "H": s["H"],
                   "2B": s["doubles"], "3B": s["triples"], "HR": s["HR"],
                   "BA": fmt(s["BA"]), "SLG": fmt(s["SLG"])}
        else:
            row = {"": "Total", "PA": s["PA"], "AB": s["AB"], "H": s["H"],
                   "2B": s["doubles"], "3B": s["triples"], "HR": s["HR"],
                   "BB": s["BB"], "K": s["K"], "HBP": s["HBP"],
                   "BA": fmt(s["BA"]), "OBP": fmt(s["OBP"]),
                   "SLG": fmt(s["SLG"]), "OPS": fmt(s["OPS"]),
                   "wOBA": fmt(s["wOBA"])}

        out = pd.DataFrame([row])
        thresholds = batter_percentile_thresholds()

        STAT_COLS = ["BA", "OBP", "SLG", "OPS", "wOBA"]
        GREEN = "font-weight:700 !important;font-size:13px !important;color:#1a7a3a !important;background-color:rgba(39,174,96,0.15) !important;"
        YELLOW = "font-weight:700 !important;font-size:13px !important;color:#8a6d00 !important;background-color:rgba(241,196,15,0.18) !important;"
        RED = "font-weight:700 !important;font-size:13px !important;color:#c0392b !important;background-color:rgba(231,76,60,0.12) !important;"
        DEFAULT = "font-weight:700;font-size:13px;color:#222;"

        def color_stats(col):
            if col.name not in STAT_COLS or col.name not in thresholds:
                return [DEFAULT for _ in col]
            p33 = thresholds[col.name]["p33"]
            p67 = thresholds[col.name]["p67"]
            styles = []
            for v in col:
                try:
                    val = float(str(v).replace("—", "nan"))
                except ValueError:
                    val = float("nan")
                if pd.isna(val):
                    styles.append(DEFAULT)
                elif val >= p67:
                    styles.append(GREEN)
                elif val >= p33:
                    styles.append(YELLOW)
                else:
                    styles.append(RED)
            return styles

        return (
            out.style
            .hide(axis="index")
            .apply(color_stats, axis=0)
        )

    # ── per-pitch summary table (same style as pitcher usage_table) ─────────
    @output
    @render.table
    def batter_pitch_table():
        data = batter_data()
        bid  = input.player() if input.player_type() == "batter" else None
        if data is None or data.empty or not bid:
            return pd.DataFrame(columns=["Pitch type"])

        sel     = batter_selected_pitch()
        order   = batter_pitch_order()

        SWING_EVENTS   = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        CONTACT_EVENTS = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}

        def fmt_ba(v):
            if v is None:
                return "—"
            if v >= 1.0:
                return f"{v:.3f}"
            return f".{int(round(v * 1000)):03d}"
        def pct_str(n, d):
            return f"{n / d * 100:.1f}%" if d > 0 else "—"

        rows = []
        for pt in order:
            ptd = data[data[PITCH_TYPE_COL].astype(str).str.strip() == pt]
            n = len(ptd)
            if n == 0:
                continue
            avg_velo = ptd["RelSpeed"].dropna().mean() if "RelSpeed" in ptd.columns else None
            calls = ptd["PitchCall"].astype(str).str.strip() if "PitchCall" in ptd.columns else pd.Series(dtype=str)
            n_swing = calls.isin(SWING_EVENTS).sum()
            n_contact = calls.isin(CONTACT_EVENTS).sum()
            n_whiff = (calls == "StrikeSwinging").sum()

            loc_h = pd.to_numeric(ptd["PlateLocHeight"], errors="coerce") if "PlateLocHeight" in ptd.columns else pd.Series(dtype=float)
            loc_s = pd.to_numeric(ptd["PlateLocSide"], errors="coerce") if "PlateLocSide" in ptd.columns else pd.Series(dtype=float)
            in_zone = (loc_h >= ZONE_BOTTOM) & (loc_h <= ZONE_TOP) & (loc_s >= ZONE_LEFT) & (loc_s <= ZONE_RIGHT)
            out_zone = ~in_zone & loc_h.notna() & loc_s.notna()
            n_out = out_zone.sum()
            n_chase = (out_zone & calls.isin(SWING_EVENTS)).sum()

            results = ptd["PlayResult"].astype(str).str.strip() if "PlayResult" in ptd.columns else pd.Series(dtype=str)
            singles = (results == "Single").sum()
            doubles = (results == "Double").sum()
            triples = (results == "Triple").sum()
            hr = (results == "HomeRun").sum()
            k_count = (results == "Strikeout").sum()
            h = singles + doubles + triples + hr
            ab_approx = h + (results == "Out").sum() + (results == "FieldersChoice").sum() + (results == "Error").sum() + k_count
            tb = singles + 2 * doubles + 3 * triples + 4 * hr

            rows.append({
                "Pitch type": pt,
                "Seen": n,
                "Avg Velo": f"{avg_velo:.1f}" if avg_velo and not math.isnan(avg_velo) else "—",
                "Swing %": pct_str(n_swing, n),
                "Whiff %": pct_str(n_whiff, n_swing),
                "Chase %": pct_str(n_chase, n_out),
                "Contact %": pct_str(n_contact, n_swing),
                "1B": singles, "2B": doubles, "3B": triples, "HR": hr, "K": k_count,
                "BA": fmt_ba(h / ab_approx if ab_approx > 0 else None),
                "SLG": fmt_ba(tb / ab_approx if ab_approx > 0 else None),
            })

        if not rows:
            return pd.DataFrame(columns=["Pitch type"])

        out = pd.DataFrame(rows)

        def highlight_col(col):
            is_sel_row = (
                out["Pitch type"].astype(str).str.strip() == str(sel).strip()
                if sel else pd.Series(False, index=out.index)
            )
            # Selected row style — warm yellow bg + gold left border on first col
            # !important is required because .usage-table-wrap CSS uses !important
            # on zebra-stripe background and would otherwise override inline styles.
            selected_style_main = (
                "font-size:18px !important;font-weight:800 !important;color:#000000 !important;"
                "background-color:#fff8e1 !important;"
                "border-left:4px solid #DDB945 !important;"
                "padding:12px 10px !important;"
            )
            selected_style_other = (
                "font-size:18px !important;font-weight:800 !important;color:#000000 !important;"
                "background-color:#fff8e1 !important;"
                "padding:12px 10px !important;"
            )
            normal_style_main = "font-size:12.5px;color:#444444;"
            normal_style_other = "font-size:12px;color:#555555;"

            if col.name == "Pitch type":
                return [selected_style_main if v else normal_style_main for v in is_sel_row]
            return [selected_style_other if v else normal_style_other for v in is_sel_row]

        return (
            out.style
            .hide(axis="index")
            .apply(highlight_col, axis=0)
        )

    # ── helpers shared by both zone charts ─────────────────────────────────
    def _batter_zone_data():
        """Filter batter data for the selected player + pitch type, with location."""
        data = batter_data()
        bid = input.player() if input.player_type() == "batter" else None
        if data is None or data.empty or not bid:
            return None, ""
        d = data[data["BatterId"].astype(str) == str(bid)].copy()
        sel = batter_selected_pitch()
        if sel and PITCH_TYPE_COL in d.columns:
            d = d[d[PITCH_TYPE_COL].astype(str).str.strip() == sel]
        for c in ["PlateLocSide", "PlateLocHeight"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        pitcher_hand = ""
        if "PitcherThrows" in d.columns:
            pt = d["PitcherThrows"].dropna().astype(str).str.strip()
            if not pt.empty:
                val = pt.iloc[0].lower()
                pitcher_hand = "RHP" if "right" in val else "LHP" if "left" in val else ""
        if "PitchCall" not in d.columns:
            d["PitchCall"] = ""
        if "PlayResult" not in d.columns:
            d["PlayResult"] = ""
        return d, pitcher_hand

    def _draw_empty_zone(ax):
        zw = (ZONE_RIGHT - ZONE_LEFT) / 3
        zh = (ZONE_TOP - ZONE_BOTTOM) / 3
        for r in range(3):
            for c in range(3):
                ax.add_patch(Rectangle(
                    (ZONE_LEFT + c * zw, ZONE_BOTTOM + r * zh), zw, zh,
                    facecolor="white", edgecolor="#aaa", linewidth=1.2, zorder=2,
                ))
        ax.add_patch(Rectangle(
            (ZONE_LEFT, ZONE_BOTTOM), ZONE_RIGHT - ZONE_LEFT, ZONE_TOP - ZONE_BOTTOM,
            fill=False, linewidth=2.5, edgecolor="#222", zorder=5,
        ))

    def _style_zone_ax(ax):
        ax.set_xlim(ZONE_LEFT - 1.0, ZONE_RIGHT + 1.0)
        ax.set_ylim(ZONE_BOTTOM - 0.6, ZONE_TOP + 0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── out rate by zone heatmap ─────────────────────────────────────────────
    @output
    @render.plot
    def batter_out_rate_plot():
        d, pitcher_hand = _batter_zone_data()

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#f7f7f7")
        ax.set_box_aspect(1)

        zw = (ZONE_RIGHT - ZONE_LEFT) / 3
        zh = (ZONE_TOP - ZONE_BOTTOM) / 3
        col_edges = [ZONE_LEFT + i * zw for i in range(4)]
        row_edges = [ZONE_BOTTOM + i * zh for i in range(4)]

        OUT_RESULTS = {"Out", "FieldersChoice", "Strikeout", "Sacrifice",
                       "SacrificeFly", "SacrificeBunt", "Error"}
        PA_RESULTS = {"Single", "Double", "Triple", "HomeRun", "Out",
                      "FieldersChoice", "Strikeout", "Walk", "HitByPitch",
                      "SacrificeFly", "SacrificeBunt", "Sacrifice",
                      "Error", "CatcherInterference"}

        total_pa = 0
        total_outs = 0

        if d is not None and not d.empty:
            pc = d["PitchCall"].astype(str).str.strip()
            pr = d["PlayResult"].astype(str).str.strip()

            # Only count terminal pitches (PA-ending) for out rate
            is_terminal = pr.isin(PA_RESULTS) | pc.eq("HitByPitch")

            # Shadow zone background (green, matching old deployed style)
            ax.add_patch(Rectangle(
                (ZONE_LEFT - 0.35, ZONE_BOTTOM - 0.30),
                (ZONE_RIGHT - ZONE_LEFT) + 0.70, (ZONE_TOP - ZONE_BOTTOM) + 0.60,
                facecolor="#edf4ee", edgecolor="none", zorder=0,
            ))

            for row_i in range(3):
                for col_i in range(3):
                    x0, x1 = col_edges[col_i], col_edges[col_i + 1]
                    y0, y1 = row_edges[row_i], row_edges[row_i + 1]
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

                    in_cell = d["PlateLocSide"].between(x0, x1) & d["PlateLocHeight"].between(y0, y1)
                    terminal_in_cell = in_cell & is_terminal
                    n_pa = int(terminal_in_cell.sum())
                    n_outs = int((terminal_in_cell & pr.isin(OUT_RESULTS)).sum())
                    total_pa += n_pa
                    total_outs += n_outs

                    if n_pa == 0:
                        cell_bg = "#f5f5f5"
                    else:
                        out_pct = n_outs / n_pa
                        if out_pct >= 0.75:
                            cell_bg = "#c0392b"
                        elif out_pct >= 0.50:
                            cell_bg = "#f0b000"
                        elif out_pct >= 0.25:
                            cell_bg = "#f5dfb3"
                        else:
                            cell_bg = "#ffffff"

                    ax.add_patch(Rectangle(
                        (x0, y0), zw, zh,
                        facecolor=cell_bg, edgecolor="#999", linewidth=1.2, zorder=2,
                    ))

                    if n_pa == 0:
                        ax.text(cx, cy, "—", ha="center", va="center",
                                fontsize=13, color="#aaa", zorder=3)
                        continue

                    out_pct = n_outs / n_pa
                    pct_txt = f"{int(round(out_pct * 100))}%"
                    frac_txt = f"{n_outs}/{n_pa}"
                    txt_color = "#ffffff" if out_pct >= 0.40 else "#222222"

                    ax.text(cx, cy + zh * 0.16, pct_txt,
                            ha="center", va="center", fontsize=16,
                            fontweight="bold", color=txt_color, zorder=3)
                    ax.text(cx, cy - zh * 0.10, frac_txt,
                            ha="center", va="center", fontsize=10,
                            color="#ffe0db" if out_pct >= 0.60 else "#666", zorder=3)

            ax.add_patch(Rectangle(
                (ZONE_LEFT, ZONE_BOTTOM), ZONE_RIGHT - ZONE_LEFT, ZONE_TOP - ZONE_BOTTOM,
                fill=False, linewidth=2.5, edgecolor="#222", zorder=5,
            ))
        else:
            _draw_empty_zone(ax)

        # Subtitle
        out_pct_total = f"{int(round(total_outs / total_pa * 100))}%" if total_pa > 0 else "—"
        vs_str = f" | vs {pitcher_hand}" if pitcher_hand else ""
        ax.text(0.5, 1.03,
                f"Out Rate: {out_pct_total} | In-Zone PA: {total_pa}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color="#555")

        _style_zone_ax(ax)
        fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.92)
        plt.close(fig)
        return fig

    # ── individual pitch results scatter (plotly for zoom) ──────────────────
    @output
    @render.ui
    def batter_pitch_results_plot():
        d, pitcher_hand = _batter_zone_data()

        fig = go.Figure()

        RESULT_TRACES = [
            ("Swing & Miss",  {"calls": {"StrikeSwinging"},
                               "symbol": "x", "color": "#E24B4A", "size": 10}),
            ("Foul",          {"calls": {"FoulBallFieldable", "FoulBallNotFieldable"},
                               "symbol": "triangle-up", "color": "#7B68AE", "size": 9}),
            ("Called Strike", {"calls": {"StrikeCalled"},
                               "symbol": "square", "color": "#E67E22", "size": 8}),
            ("Ball",          {"calls": {"BallCalled", "Ball"},
                               "symbol": "circle", "color": "#aaaaaa", "size": 7}),
        ]
        HIT_RESULTS = {"Single", "Double", "Triple", "HomeRun"}
        OUT_RESULTS = {"Out", "FieldersChoice", "Error", "Sacrifice",
                       "SacrificeFly", "SacrificeBunt"}

        total_pitches = 0

        if d is not None and not d.empty:
            pc = d["PitchCall"].astype(str).str.strip()
            pr = d["PlayResult"].astype(str).str.strip()
            total_pitches = len(d)

            # Strike zone
            fig.add_shape(type="rect",
                x0=ZONE_LEFT, y0=ZONE_BOTTOM, x1=ZONE_RIGHT, y1=ZONE_TOP,
                line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)",
            )
            # Home plate
            hp_x = [-0.35, -0.35, 0, 0.35, 0.35]
            hp_y = [1.2, 1.0, 0.9, 1.0, 1.2]
            fig.add_trace(go.Scatter(
                x=hp_x + [hp_x[0]], y=hp_y + [hp_y[0]],
                mode="lines", fill="toself",
                fillcolor="rgba(255,255,255,0.8)",
                line=dict(color="#555", width=1),
                showlegend=False, hoverinfo="skip",
            ))

            # Non-InPlay traces
            for label, sty in RESULT_TRACES:
                mask = pc.isin(sty["calls"])
                if not mask.any():
                    continue
                g = d[mask]
                fig.add_trace(go.Scatter(
                    x=g["PlateLocSide"], y=g["PlateLocHeight"],
                    mode="markers", name=label,
                    hovertemplate=f"{label}<br>Side: %{{x:.2f}}<br>Height: %{{y:.2f}}<extra></extra>",
                    marker=dict(
                        size=sty["size"], color=sty["color"],
                        symbol=sty["symbol"], opacity=0.82,
                        line=dict(color="white", width=0.5),
                    ),
                ))

            # InPlay — split hits vs outs
            in_play = d[pc.eq("InPlay")]
            if not in_play.empty:
                ip_pr = in_play["PlayResult"].astype(str).str.strip()
                hits = in_play[ip_pr.isin(HIT_RESULTS)]
                outs = in_play[ip_pr.isin(OUT_RESULTS)]

                if not hits.empty:
                    fig.add_trace(go.Scatter(
                        x=hits["PlateLocSide"], y=hits["PlateLocHeight"],
                        mode="markers", name="In-Play Hit",
                        hovertemplate="Hit (%{text})<br>Side: %{x:.2f}<br>Height: %{y:.2f}<extra></extra>",
                        text=hits["PlayResult"].astype(str).str.strip(),
                        marker=dict(size=10, color="#2E8B57", symbol="diamond",
                                    opacity=0.85, line=dict(color="white", width=0.5)),
                    ))
                if not outs.empty:
                    fig.add_trace(go.Scatter(
                        x=outs["PlateLocSide"], y=outs["PlateLocHeight"],
                        mode="markers", name="In-Play Out",
                        hovertemplate="Out<br>Side: %{x:.2f}<br>Height: %{y:.2f}<extra></extra>",
                        marker=dict(size=10, color="#8B0000", symbol="diamond",
                                    opacity=0.85, line=dict(color="white", width=0.5)),
                    ))
        else:
            fig.add_shape(type="rect",
                x0=ZONE_LEFT, y0=ZONE_BOTTOM, x1=ZONE_RIGHT, y1=ZONE_TOP,
                line=dict(color="#222", width=2), fillcolor="rgba(0,0,0,0)",
            )
            fig.add_annotation(text="No data", x=0, y=2.5,
                               showarrow=False, font=dict(size=14, color="#888"))

        vs_str = f" | vs {pitcher_hand}" if pitcher_hand else ""
        fig.update_layout(
            title=dict(text=f"n = {total_pitches}{vs_str}", font=dict(size=12, color="#555"),
                       x=0.5, xanchor="center"),
            paper_bgcolor="#f7f7f7", plot_bgcolor="#f7f7f7",
            height=520, dragmode="zoom",
            legend=dict(
                orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
                font=dict(size=10), bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#ccc", borderwidth=1,
                itemclick="toggleothers",
                itemdoubleclick="toggle",
            ),
            margin=dict(l=10, r=10, t=35, b=50),
        )
        fig.update_xaxes(
            range=[ZONE_LEFT - 1.0, ZONE_RIGHT + 1.0],
            showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
            showticklabels=False,
        )
        fig.update_yaxes(
            range=[ZONE_BOTTOM - 0.6, ZONE_TOP + 0.8],
            showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
            scaleanchor="x", scaleratio=1,
            showticklabels=False,
        )

        return ui.HTML(fig.to_html(
            full_html=False, include_plotlyjs=False,
            config={
                "displayModeBar": True, "displaylogo": False, "scrollZoom": True,
                "modeBarButtonsToRemove": [
                    "toImage", "select2d", "lasso2d",
                    "hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines",
                ],
            },
        ))

    # ── scout insights ──────────────────────────────────────────────────────
    @output
    @render.ui
    def batter_scout_insights():
        data = batter_data()
        bid = input.player() if input.player_type() == "batter" else None
        if data is None or data.empty or not bid:
            return ui.div()

        d_all = data[data["BatterId"].astype(str) == str(bid)].copy()
        sel = batter_selected_pitch()
        if sel and PITCH_TYPE_COL in d_all.columns:
            d_all = d_all[d_all[PITCH_TYPE_COL].astype(str).str.strip() == sel]

        if d_all.empty:
            return ui.div()

        SWING_EV = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        CONTACT_EV = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        MIN_PITCHES = 4

        def _badge_item(badge, color, loc, desc):
            return ui.tags.div(
                ui.tags.span(badge, style=(
                    f"display:inline-block;background:{color};color:#fff;"
                    "font-size:9px;font-weight:800;letter-spacing:0.5px;"
                    "padding:2px 8px;border-radius:3px;margin-right:8px;"
                    "white-space:nowrap;"
                )),
                ui.tags.span(f"{loc}: ", style="font-weight:700;font-size:12px;color:#222;margin-right:2px;"),
                ui.tags.span(desc, style="font-size:12px;line-height:1.5;color:#444;"),
                style="display:flex;align-items:center;flex-wrap:wrap;padding:6px 0;",
            )

        def _section(title, items):
            if not items:
                return None
            return ui.tags.div(
                ui.tags.div(title, style=(
                    "font-size:11px;font-weight:700;color:#888;text-transform:uppercase;"
                    "letter-spacing:0.5px;padding-bottom:4px;margin-bottom:4px;"
                    "border-bottom:1px solid #eee;"
                )),
                *items,
                style="margin-bottom:24px;padding-bottom:12px;border-bottom:2px solid #e8e8e8;",
            )

        sec_location = None
        sec_spray = None
        sec_discipline = None
        sec_matchups = None
        sec_count = None

        # ── Pitch Location insights (max 2) ──
        d = d_all.copy()
        for c in ["PlateLocSide", "PlateLocHeight"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"])

        loc_candidates = []  # (score, badge, color, loc, desc)

        if not d.empty:
            calls = d["PitchCall"].astype(str).str.strip() if "PitchCall" in d.columns else pd.Series("", index=d.index)
            zw = (ZONE_RIGHT - ZONE_LEFT) / 3
            zh = (ZONE_TOP - ZONE_BOTTOM) / 3
            ROW_LABELS = ["lower", "middle", "upper"]
            COL_LABELS = ["inside", "middle", "outside"]

            for ri in range(3):
                for ci in range(3):
                    left = ZONE_LEFT + ci * zw
                    bot = ZONE_BOTTOM + ri * zh
                    mask = (
                        (d["PlateLocSide"] >= left) & (d["PlateLocSide"] < left + zw) &
                        (d["PlateLocHeight"] >= bot) & (d["PlateLocHeight"] < bot + zh)
                    )
                    n = mask.sum()
                    if n < MIN_PITCHES:
                        continue
                    cell_calls = calls[mask]
                    n_sw = cell_calls.isin(SWING_EV).sum()
                    n_ct = cell_calls.isin(CONTACT_EV).sum()
                    sw_pct = n_sw / n
                    ct_pct = n_ct / n_sw if n_sw > 0 else 0
                    loc = f"{ROW_LABELS[ri]}-{COL_LABELS[ci]}"

                    if sw_pct >= 0.70 and ct_pct >= 0.80:
                        loc_candidates.append((sw_pct * ct_pct * n, "ATTACK ZONE", "#c0392b", loc,
                            f"{sw_pct:.0%} swing rate, {ct_pct:.0%} contact. "
                            "Aggressive and making hard contact. Do not pitch here."))
                    if sw_pct >= 0.50 and ct_pct < 0.50 and n_sw >= MIN_PITCHES:
                        loc_candidates.append((sw_pct * (1 - ct_pct) * n, "STRIKEOUT ZONE", "#7B2FBE", loc,
                            f"{sw_pct:.0%} swing rate, only {ct_pct:.0%} contact. "
                            "Will commit but can't catch up. Go-to spot for the punchout."))
                    if sw_pct <= 0.30:
                        loc_candidates.append(((1 - sw_pct) * n, "SAFE ZONE", "#1D9E75", loc,
                            f"Only {sw_pct:.0%} swing rate on {n} pitches. "
                            "Very passive here. Live here early in counts to steal called strikes."))

            PAD_H, PAD_V = 0.5, 0.5
            for label, ol, orr, ob, ot in [
                ("above the zone", ZONE_LEFT, ZONE_RIGHT, ZONE_TOP, ZONE_TOP + PAD_V),
                ("below the zone", ZONE_LEFT, ZONE_RIGHT, ZONE_BOTTOM - PAD_V, ZONE_BOTTOM),
                ("inside",  ZONE_LEFT - PAD_H, ZONE_LEFT, ZONE_BOTTOM, ZONE_TOP),
                ("outside", ZONE_RIGHT, ZONE_RIGHT + PAD_H, ZONE_BOTTOM, ZONE_TOP),
            ]:
                mask = (
                    (d["PlateLocSide"] >= ol) & (d["PlateLocSide"] < orr) &
                    (d["PlateLocHeight"] >= ob) & (d["PlateLocHeight"] < ot)
                )
                n = mask.sum()
                if n < MIN_PITCHES:
                    continue
                cell_calls = calls[mask]
                n_sw = cell_calls.isin(SWING_EV).sum()
                chase_pct = n_sw / n
                if chase_pct >= 0.50:
                    loc_candidates.append((chase_pct * n, "EXPAND HERE", "#E67E22", label,
                        f"{chase_pct:.0%} chase rate on {n} pitches. "
                        "Tunnel something that looks like a strike and let it run off the plate."))
                elif chase_pct <= 0.15:
                    loc_candidates.append(((1 - chase_pct) * n, "DON'T WASTE", "#2980B9", label,
                        f"Only {chase_pct:.0%} chase rate on {n} pitches. "
                        "Not expanding here. Compete with strikes, don't waste pitches."))

        loc_candidates.sort(key=lambda x: x[0], reverse=True)
        loc_items = [_badge_item(b, c, l, d) for _, b, c, l, d in loc_candidates[:2]]
        sec_location = _section("Pitch Location", loc_items)

        # ── 2. Spray Chart insights (max 2) ──
        spray_items = []
        if "Direction" in d_all.columns and "PlayResult" in d_all.columns:
            pr = d_all["PlayResult"].astype(str).str.strip()
            hits = d_all[pr.isin({"Single", "Double", "Triple", "HomeRun"})].copy()
            if len(hits) >= 5:
                dirs = pd.to_numeric(hits["Direction"], errors="coerce").dropna()
                if len(dirs) >= 5:
                    pull_pct = (dirs < -15).mean()
                    oppo_pct = (dirs > 15).mean()
                    mid_pct = ((dirs >= -15) & (dirs <= 15)).mean()
                    if pull_pct >= 0.60:
                        spray_items.append(_badge_item("PULL HEAVY", "#8B4513", "spray chart",
                            f"{pull_pct:.0%} of hits go to the pull side. "
                            "Shade the defense pull-side. Pitch away to neutralize."))
                    elif oppo_pct >= 0.50:
                        spray_items.append(_badge_item("USES ALL FIELDS", "#2E8B57", "spray chart",
                            f"{oppo_pct:.0%} of hits go opposite field. "
                            "Can't rely on pull-side shifts. Must locate to both sides of the plate."))
                    elif mid_pct >= 0.50:
                        spray_items.append(_badge_item("UP THE MIDDLE", "#555", "spray chart",
                            f"{mid_pct:.0%} of hits go up the middle. "
                            "Concentrated up the middle. Shade middle infielders accordingly."))

                    # Power location
                    xbh = d_all[pr.isin({"Double", "Triple", "HomeRun"})].copy()
                    if len(xbh) >= 3:
                        xbh_dirs = pd.to_numeric(xbh["Direction"], errors="coerce").dropna()
                        if len(xbh_dirs) >= 3:
                            xbh_pull = (xbh_dirs < -15).mean()
                            xbh_oppo = (xbh_dirs > 15).mean()
                            if xbh_pull >= 0.65:
                                spray_items.append(_badge_item("PULL POWER", "#8B0000", "extra-base hits",
                                    f"{xbh_pull:.0%} of extra-base hits go pull side. "
                                    "Power is to the pull side. Work away to limit damage."))
                            elif xbh_oppo >= 0.50:
                                spray_items.append(_badge_item("OPPO POWER", "#2E8B57", "extra-base hits",
                                    f"{xbh_oppo:.0%} of extra-base hits go opposite field. "
                                    "Generates power the other way. Tough to game plan against."))

        sec_spray = _section("Spray Chart", spray_items[:2])

        # ── 3. Plate Discipline insights (max 2) ──
        disc_items = []
        if "PitchCall" in d_all.columns:
            pc_all = d_all["PitchCall"].astype(str).str.strip()
            total_sw = pc_all.isin(SWING_EV).sum()
            total_whiff = (pc_all == "StrikeSwinging").sum()
            total_p = len(d_all)

            if total_sw >= 10:
                whiff_rate = total_whiff / total_sw
                contact_rate = 1 - whiff_rate
                if whiff_rate >= 0.25:
                    disc_items.append(_badge_item("HIGH WHIFF", "#7B2FBE", "plate discipline",
                        f"{whiff_rate:.0%} whiff rate across all swings. "
                        "Swing-and-miss in the arsenal. Can be put away with two strikes."))
                elif whiff_rate <= 0.08 and total_sw >= 20:
                    disc_items.append(_badge_item("BAT-TO-BALL", "#2E8B57", "plate discipline",
                        f"Only {whiff_rate:.0%} whiff rate on {total_sw} swings. "
                        "Rarely misses. Must locate and induce weak contact, strikeouts will be tough."))

            # Chase rate overall
            if "PlateLocSide" in d_all.columns and "PlateLocHeight" in d_all.columns:
                lh = pd.to_numeric(d_all["PlateLocHeight"], errors="coerce")
                ls = pd.to_numeric(d_all["PlateLocSide"], errors="coerce")
                iz = (lh >= ZONE_BOTTOM) & (lh <= ZONE_TOP) & (ls >= ZONE_LEFT) & (ls <= ZONE_RIGHT)
                oz = ~iz & lh.notna() & ls.notna()
                n_oz = oz.sum()
                if n_oz >= 10:
                    oz_sw = (oz & pc_all.isin(SWING_EV)).sum()
                    overall_chase = oz_sw / n_oz
                    if overall_chase >= 0.35:
                        disc_items.append(_badge_item("CHASER", "#E67E22", "plate discipline",
                            f"{overall_chase:.0%} overall chase rate on {n_oz} pitches outside the zone. "
                            "Expands the zone frequently. Use off-speed and breaking balls to exploit."))
                    elif overall_chase <= 0.12:
                        disc_items.append(_badge_item("DISCIPLINED", "#2980B9", "plate discipline",
                            f"Only {overall_chase:.0%} chase rate on {n_oz} pitches outside the zone. "
                            "Very disciplined eye. Must throw strikes to get ahead."))

        sec_discipline = _section("Plate Discipline", disc_items[:2])

        # ── 4. Pitch Outcomes (which pitches this batter struggles/thrives against) ──
        outcome_items = []
        if PITCH_TYPE_COL in d_all.columns and "PlayResult" in d_all.columns:
            pr_all = d_all["PlayResult"].astype(str).str.strip()
            pc_all2 = d_all["PitchCall"].astype(str).str.strip() if "PitchCall" in d_all.columns else pd.Series("", index=d_all.index)
            OUT_RESULTS = {"Out", "FieldersChoice", "Strikeout", "Sacrifice",
                           "SacrificeFly", "SacrificeBunt", "Error"}
            PA_RESULTS_B = {"Single", "Double", "Triple", "HomeRun", "Out", "FieldersChoice",
                          "Strikeout", "Walk", "HitByPitch", "SacrificeFly", "SacrificeBunt",
                          "Sacrifice", "Error", "CatcherInterference"}
            HIT_RESULTS_B = {"Single", "Double", "Triple", "HomeRun"}
            SW_B = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}

            is_terminal_b = pr_all.isin(PA_RESULTS_B) | pc_all2.eq("HitByPitch")
            valid_pt = d_all[is_valid_pitch_type(d_all[PITCH_TYPE_COL])].copy()

            pt_stats = []
            for pt, g in valid_pt.groupby(PITCH_TYPE_COL):
                g_pc = g["PitchCall"].astype(str).str.strip()
                g_pr = g["PlayResult"].astype(str).str.strip()
                g_term = g_pr.isin(PA_RESULTS_B) | g_pc.eq("HitByPitch")
                n_pa = int(g_term.sum())
                n_outs = int((g_term & g_pr.isin(OUT_RESULTS)).sum())
                n_hits = int((g_term & g_pr.isin(HIT_RESULTS_B)).sum())
                n_ab = n_pa - int((g_term & (g_pr.eq("Walk") | g_pc.eq("HitByPitch"))).sum())
                ba = n_hits / n_ab if n_ab > 0 else None
                out_rate = n_outs / n_pa if n_pa > 0 else None
                n_sw = int(g_pc.isin(SW_B).sum())
                n_whiff = int(g_pc.eq("StrikeSwinging").sum())
                whiff_rate = n_whiff / n_sw if n_sw > 0 else None
                pt_stats.append({"pt": pt, "n_pa": n_pa, "n_pitches": len(g),
                                 "out_rate": out_rate, "ba": ba, "whiff_rate": whiff_rate})

            qualified_b = [s for s in pt_stats if s["n_pa"] >= 5]

            if qualified_b:
                # Pitch this batter struggles against (highest out rate)
                best_out = max(qualified_b, key=lambda s: s["out_rate"] or 0)
                if best_out["out_rate"] is not None and best_out["out_rate"] >= 0.50:
                    outcome_items.append(_badge_item("STRUGGLES AGAINST", "#c0392b", best_out["pt"],
                        f"{best_out['out_rate']:.0%} out rate ({best_out['n_pa']} PA). "
                        "Attack with this pitch to get outs."))

                # Pitch this batter thrives against (highest BA)
                ba_qualified = [s for s in qualified_b if s["ba"] is not None and s["ba"] > 0]
                if ba_qualified:
                    best_ba = max(ba_qualified, key=lambda s: s["ba"])
                    if best_ba["ba"] >= 0.300:
                        ba_fmt = f"{best_ba['ba']:.3f}" if best_ba["ba"] >= 1.0 else f".{int(round(best_ba['ba']*1000)):03d}"
                        outcome_items.append(_badge_item("THRIVES AGAINST", "#1D9E75", best_ba["pt"],
                            f"{ba_fmt} BA ({best_ba['n_pa']} PA). "
                            "Avoid this pitch or relocate carefully."))

                # Most vulnerable to whiffs
                whiff_q = [s for s in pt_stats if s["whiff_rate"] is not None and s["n_pitches"] >= 10]
                if whiff_q:
                    best_whiff = max(whiff_q, key=lambda s: s["whiff_rate"])
                    if best_whiff["whiff_rate"] >= 0.20:
                        outcome_items.append(_badge_item("CAN'T CATCH UP TO", "#7B2FBE", best_whiff["pt"],
                            f"{best_whiff['whiff_rate']:.0%} whiff rate. "
                            "Use as put-away pitch in 2-strike counts."))

        sec_matchups = _section("Pitch Matchups", outcome_items[:3])

        # ── 5. Count Approach (how this batter performs in different counts) ──
        count_approach_items = []
        if "Balls" in d_all.columns and "Strikes" in d_all.columns and "PitchCall" in d_all.columns:
            b_col = pd.to_numeric(d_all["Balls"], errors="coerce")
            s_col = pd.to_numeric(d_all["Strikes"], errors="coerce")
            pc_ca = d_all["PitchCall"].astype(str).str.strip()
            pr_ca = d_all["PlayResult"].astype(str).str.strip() if "PlayResult" in d_all.columns else pd.Series("", index=d_all.index)

            CA_SW = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
            CA_OUT = {"Out", "FieldersChoice", "Strikeout", "Sacrifice", "SacrificeFly", "SacrificeBunt", "Error"}
            CA_HIT = {"Single", "Double", "Triple", "HomeRun"}
            CA_PA = {"Single", "Double", "Triple", "HomeRun", "Out", "FieldersChoice",
                     "Strikeout", "Walk", "HitByPitch", "SacrificeFly", "SacrificeBunt",
                     "Sacrifice", "Error", "CatcherInterference"}

            COUNT_GROUPS = [
                ("Hitter's Count", [(2, 0), (3, 0), (3, 1)], "#1D9E75"),
                ("Two Strikes", [(0, 2), (1, 2), (2, 2)], "#c0392b"),
                ("First Pitch", [(0, 0)], "#185FA5"),
                ("Even Count", [(1, 1)], "#E67E22"),
            ]

            for label, counts, color in COUNT_GROUPS:
                mask = pd.Series(False, index=d_all.index)
                for b, s in counts:
                    mask = mask | (b_col.eq(b) & s_col.eq(s))

                cd = d_all[mask]
                if len(cd) < 8:
                    continue

                cd_pc = cd["PitchCall"].astype(str).str.strip()
                cd_pr = cd["PlayResult"].astype(str).str.strip() if "PlayResult" in cd.columns else pd.Series("", index=cd.index)
                cd_terminal = cd_pr.isin(CA_PA) | cd_pc.eq("HitByPitch")

                n_pitches = len(cd)
                n_sw = int(cd_pc.isin(CA_SW).sum())
                sw_rate = n_sw / n_pitches if n_pitches > 0 else 0

                n_pa = int(cd_terminal.sum())
                n_hits = int((cd_terminal & cd_pr.isin(CA_HIT)).sum())
                n_outs = int((cd_terminal & cd_pr.isin(CA_OUT)).sum())
                n_ab = n_pa - int((cd_terminal & (cd_pr.eq("Walk") | cd_pc.eq("HitByPitch"))).sum())
                ba = n_hits / n_ab if n_ab > 0 else None

                # Chase rate in this count
                if "PlateLocSide" in cd.columns and "PlateLocHeight" in cd.columns:
                    ls = pd.to_numeric(cd["PlateLocSide"], errors="coerce")
                    lh = pd.to_numeric(cd["PlateLocHeight"], errors="coerce")
                    oz = ~(ls.between(ZONE_LEFT, ZONE_RIGHT) & lh.between(ZONE_BOTTOM, ZONE_TOP)) & ls.notna() & lh.notna()
                    n_oz = int(oz.sum())
                    chase_rate = int((oz & cd_pc.isin(CA_SW)).sum()) / n_oz if n_oz > 5 else None
                else:
                    chase_rate = None

                # Build insight text
                parts = []
                if ba is not None:
                    ba_fmt = f"{ba:.3f}" if ba >= 1.0 else f".{int(round(ba*1000)):03d}"
                    parts.append(f"{ba_fmt} BA")
                parts.append(f"{sw_rate:.0%} swing rate")
                if chase_rate is not None:
                    parts.append(f"{chase_rate:.0%} chase rate")
                stat_line = ", ".join(parts)

                # Coaching advice based on count group
                count_str = " / ".join(f"{b}-{s}" for b, s in counts)
                if label == "Hitter's Count":
                    if sw_rate >= 0.50 and ba is not None and ba >= 0.300:
                        advice = "Aggressive and productive — keep hunting your pitch."
                    elif sw_rate < 0.30:
                        advice = "Not aggressive enough — this is your count, look to drive."
                    else:
                        advice = "Be selective but ready to attack a good pitch."
                elif label == "Two Strikes":
                    if chase_rate is not None and chase_rate >= 0.35:
                        advice = "Chasing too much — shorten up and protect the zone."
                    elif chase_rate is not None and chase_rate <= 0.15:
                        advice = "Disciplined with two strikes — forces pitcher to throw strikes."
                    else:
                        advice = "Battle mode — fight off tough pitches, don't give away at-bats."
                elif label == "First Pitch":
                    if sw_rate >= 0.40 and ba is not None and ba >= 0.300:
                        advice = "Aggressive first-pitch hitter and it's working — keep jumping early."
                    elif sw_rate >= 0.40 and (ba is None or ba < 0.200):
                        advice = "Swinging first pitch but not getting results — be more selective early."
                    elif sw_rate < 0.20:
                        advice = "Very passive first pitch — could be missing hittable pitches early."
                    else:
                        advice = "Balanced first-pitch approach."
                else:  # Even
                    advice = "Key count — the at-bat often turns here."

                count_approach_items.append(_badge_item(label.upper(), color, count_str,
                    f"{stat_line}. {advice} ({n_pitches} pitches)"))

        sec_count = _section("Count Approach", count_approach_items[:4])

        # Assemble in priority order: actionable first, then summaries
        sections = [s for s in [
            sec_count,       # Count Approach (most actionable)
            sec_matchups,    # Pitch Matchups (actionable)
            sec_location,    # Pitch Location (visual summary)
            sec_spray,       # Spray Chart (visual summary)
            sec_discipline,  # Plate Discipline (visual summary)
        ] if s is not None]

        if not sections:
            return ui.div(
                ui.tags.span("Insufficient data for scouting insights.",
                             style="font-size:12px;color:#888;font-style:italic;"),
                style="padding:12px 16px;",
            )

        return ui.div(*sections, style="padding:12px 16px;")

    # ── spray chart ─────────────────────────────────────────────────────────
    @output
    @render.plot
    def batter_spray_plot():
        data = batter_data()
        bid  = input.player() if input.player_type() == "batter" else None
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        fig.patch.set_facecolor("#f7f7f7"); ax.set_facecolor("#f7f7f7")
        angles = np.linspace(np.radians(45), np.radians(135), 100)
        OF, IF = 200, 90
        ox, oy = OF*np.cos(angles), OF*np.sin(angles)
        ix, iy = IF*np.cos(angles), IF*np.sin(angles)
        ax.fill(np.append(ox,[0]), np.append(oy,[0]), color="#e8f5e9", zorder=1)
        ax.plot(np.append(ox,[0,ox[0]]), np.append(oy,[0,oy[0]]), color="#aaa", lw=1, zorder=2)
        ax.fill(np.append(ix,[0]), np.append(iy,[0]), color="#c8e6c9", zorder=1)
        ax.plot(np.append(ix,[0,ix[0]]), np.append(iy,[0,iy[0]]), color="#aaa", lw=0.8, zorder=2)
        for ang in [45,135]:
            r = np.radians(ang)
            ax.plot([0,OF*np.cos(r)],[0,OF*np.sin(r)], color="#888", lw=1, ls="--", zorder=2)
        ax.plot(0,0,"s",color="white",markersize=7,markeredgecolor="#555",zorder=5)
        HIT = {"Single":"#1D9E75","Double":"#378ADD","Triple":"#7B2FBE","HomeRun":"#BA7517"}
        if data is not None and not data.empty and bid:
            d = data[data["BatterId"].astype(str) == str(bid)].copy()
            sel = batter_selected_pitch()
            if sel and PITCH_TYPE_COL in d.columns:
                d = d[d[PITCH_TYPE_COL].astype(str).str.strip() == sel]
            for c in ["ExitSpeed","Direction"]:
                if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
            pr_s = d["PlayResult"].astype(str).str.strip() if "PlayResult" in d.columns else pd.Series("",index=d.index)
            inp  = d[pr_s.isin(set(HIT)|{"Out","FieldersChoice","Error"})].dropna(subset=["Direction"])
            inp = inp.copy()
            inp["_ang"]  = np.radians(90 - inp["Direction"].astype(float))
            inp["_ev"]   = inp["ExitSpeed"].fillna(70).astype(float)
            inp["_dist"] = inp["_ev"].apply(lambda ev: max(min(ev * 1.5, OF - 5), IF - 10))
            inp["_pr"]   = inp["PlayResult"].astype(str).str.strip()
            inp["_color"] = inp["_pr"].map(lambda pr: HIT.get(pr, "#D85A30"))
            inp["_size"]  = inp["_pr"].apply(lambda pr: 50 if pr in HIT else 35)
            for color, grp in inp.groupby("_color", sort=False):
                ax.scatter(
                    grp["_dist"] * np.cos(grp["_ang"]),
                    grp["_dist"] * np.sin(grp["_ang"]),
                    s=grp["_size"].values,
                    color=color, alpha=0.82, edgecolors="none", zorder=4,
                )
        ax.set_xlim(-220,220); ax.set_ylim(-20,230)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        leg = fig.legend(
            handles=[
                plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=c,markersize=6,label=l)
                for l,c in [("1B","#1D9E75"),("2B","#378ADD"),
                            ("3B","#7B2FBE"),("HR","#BA7517"),("Out","#D85A30")]
            ],
            loc="lower center",
            ncol=5,
            fontsize=7,
            frameon=True,
            fancybox=True,
            edgecolor="#ddd",
            facecolor="white",
            framealpha=1.0,
            handletextpad=0.3,
            columnspacing=0.8,
            borderpad=0.4,
        )
        leg.get_frame().set_linewidth(0.8)
        fig.subplots_adjust(top=0.96, bottom=0.12, left=0.04, right=0.96)

        plt.close(fig)
        return fig

    # ── exit velo vs launch angle ────────────────────────────────────────────
    @output
    @render.plot
    def batter_ev_la_plot():
        data = batter_data()
        bid = input.player() if input.player_type() == "batter" else None

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#f7f7f7")
        ax.set_box_aspect(1)

        ax.add_patch(Rectangle(
            (98, 26), 22, 4,
            color="#e8f5e9",
            zorder=1,
            linewidth=1.2,
            edgecolor="#2e7d32",
            linestyle="--",
        ))
        ax.text(
            109, 28, "Barrel zone",
            ha="center", va="center",
            fontsize=8, color="#2e7d32",
            fontweight="bold", zorder=2,
        )

        HT = {
            "GroundBall": "#D85A30",
            "LineDrive": "#1D9E75",
            "FlyBall": "#378ADD",
            "Popup": "#888780",
        }

        plotted = False
        if data is not None and not data.empty and bid:
            d = data[data["BatterId"].astype(str) == str(bid)]
            sel = batter_selected_pitch()
            if sel and PITCH_TYPE_COL in d.columns:
                d = d[d[PITCH_TYPE_COL].astype(str).str.strip() == sel]
            for c in ["ExitSpeed", "Angle"]:
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.dropna(subset=["ExitSpeed", "Angle"])
            d = d[d["ExitSpeed"] > 0]

            if not d.empty:
                plotted = True
                d["_color"] = d["TaggedHitType"].astype(str).str.strip().map(
                    lambda ht: HT.get(ht, "#aaa")
                )
                for color, grp in d.groupby("_color", sort=False):
                    ax.scatter(
                        grp["ExitSpeed"], grp["Angle"],
                        s=38, color=color, alpha=0.82,
                        edgecolors="white", linewidth=0.4, zorder=3,
                    )


        ax.axhline(0, color="#ccc", lw=0.8, ls="--")
        ax.set_xlim(20, 130)
        ax.set_ylim(-40, 90)
        ax.set_xlabel("Exit velocity (mph)", fontsize=9)
        ax.set_ylabel("Launch angle (°)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")

        if not plotted:
            ax.text(
                0.5, 0.5, "No exit velocity data",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=10, color="#888",
            )

        # Legend below x-axis label
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=6, label=l)
            for l, c in [("GB", "#D85A30"), ("LD", "#1D9E75"),
                         ("FB", "#378ADD"), ("PU", "#888780")]
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            fontsize=7,
            frameon=True,
            fancybox=True,
            edgecolor="#ddd",
            facecolor="white",
            framealpha=1.0,
            ncol=4,
            handletextpad=0.3,
            columnspacing=0.8,
            borderpad=0.4,
        )

        fig.subplots_adjust(top=0.96, bottom=0.18, left=0.14, right=0.96)

        plt.close(fig)
        return fig

    # ── exit velo distribution (batting practice only) ───────────────────────
    @output
    @render.plot
    def batter_ev_dist_plot():
        data = batter_data()
        bid = input.player() if input.player_type() == "batter" else None

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#f7f7f7")
        ax.set_box_aspect(1)

        BUCKETS = [
            (50, 60, "#B5D4F4"),
            (60, 70, "#378ADD"),
            (70, 80, "#185FA5"),
            (80, 90, "#378ADD"),
            (90, 200, "#BA7517"),
        ]
        LABELS = ["50–60", "60–70", "70–80", "80–90", "90+"]
        counts = [0] * 5

        if data is not None and not data.empty and bid:
            d = data[data["BatterId"].astype(str) == str(bid)].copy()
            sel = batter_selected_pitch()
            if sel and PITCH_TYPE_COL in d.columns:
                d = d[d[PITCH_TYPE_COL].astype(str).str.strip() == sel]
            if "ExitSpeed" in d.columns:
                ev = pd.to_numeric(d["ExitSpeed"], errors="coerce").dropna()
                ev = ev[ev > 0]
                for i, (lo, hi, _) in enumerate(BUCKETS):
                    counts[i] = int(((ev >= lo) & (ev < hi)).sum())

        xs = np.arange(5)
        bars = ax.bar(xs, counts, color=[c for _, _, c in BUCKETS], width=0.62, zorder=2)

        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(cnt),
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                )

        ax.set_xticks(xs)
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_ylabel("Hit count", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")

        ax.text(
            0.98, 0.97,
            "Amber = 90+ mph hard contact",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7.5, color="#BA7517",
        )

        fig.subplots_adjust(top=0.96, bottom=0.14, left=0.14, right=0.96)

        plt.close(fig)
        return fig

    # ── plate discipline radar (scrimmage / live only) ───────────────────────
    @output
    @render.plot
    def batter_radar_plot():
        data = batter_data()
        bid = input.player() if input.player_type() == "batter" else None

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#f7f7f7")

        LABELS = ["Zone\nSw%", "Contact%", "O-Contact%", "Chase%", "Whiff%", "Zone%"]
        N = len(LABELS)
        angles = [n / N * 2 * np.pi for n in range(N)] + [0]
        values = [0.0] * N

        if data is not None and not data.empty and bid:
            d = data[data["BatterId"].astype(str) == str(bid)].copy()
            sel = batter_selected_pitch()
            if sel and PITCH_TYPE_COL in d.columns:
                d = d[d[PITCH_TYPE_COL].astype(str).str.strip() == sel]
            pc = d["PitchCall"].astype(str).str.strip() if "PitchCall" in d.columns else pd.Series("", index=d.index)

            SW = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
            CT = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}

            if "PlateLocSide" in d.columns and "PlateLocHeight" in d.columns:
                d["PlateLocSide"] = pd.to_numeric(d["PlateLocSide"], errors="coerce")
                d["PlateLocHeight"] = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
                d["iz"] = (
                    d["PlateLocSide"].between(ZONE_LEFT, ZONE_RIGHT)
                    & d["PlateLocHeight"].between(ZONE_BOTTOM, ZONE_TOP)
                )
            else:
                d["iz"] = False

            d["sw"] = pc.isin(SW)
            d["ct"] = pc.isin(CT)
            d["wh"] = pc.eq("StrikeSwinging")

            n = len(d)
            iz = d["iz"]
            oz = ~iz
            zp = int(iz.sum())
            op = int(oz.sum())
            ts = int(d["sw"].sum())
            oz_sw = int((d["sw"] & oz).sum())

            values = [
                int((d["sw"] & iz).sum()) / zp if zp > 0 else 0,
                int(d["ct"].sum()) / ts if ts > 0 else 0,
                int((d["ct"] & oz).sum()) / oz_sw if oz_sw > 0 else 0,
                oz_sw / op if op > 0 else 0,
                int(d["wh"].sum()) / ts if ts > 0 else 0,
                zp / n if n > 0 else 0,
            ]

        vals = values + [values[0]]

        for r in [0.25, 0.50, 0.75, 1.0]:
            ax.plot(angles, [r] * (N + 1), color="#ddd", lw=0.6, ls="--", zorder=1)

        ax.plot(angles, vals, color="#185FA5", lw=2, zorder=3)
        ax.fill(angles, vals, color="#378ADD", alpha=0.18, zorder=2)
        ax.scatter(
            angles[:-1], values,
            s=42, color="#185FA5",
            zorder=4, edgecolors="white", lw=0.8,
        )

        for ang, val in zip(angles[:-1], values):
            ax.text(
                ang,
                min(val + 0.13, 1.10),
                f"{val * 100:.0f}%",
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="#185FA5", zorder=5,
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.tick_params(axis="x", pad=12)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.50, 0.75])
        ax.set_yticklabels(["25%", "50%", "75%"], fontsize=7, color="#aaa")
        ax.spines["polar"].set_visible(False)

        fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.08)

        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # HitTrax (session-aggregated batter data)
    # ------------------------------------------------------------------
    @reactive.calc
    def hittrax_data():
        """Filtered HitTrax rows for the selected player + date range."""
        if HITTRAX_DF.empty:
            return None
        player = input.player()
        if not player:
            return None
        start = input.date_start()
        end = input.date_end()
        if start is None or end is None:
            return None

        d = HITTRAX_DF[
            (HITTRAX_DF["Player"] == player)
            & (HITTRAX_DF["DateOnly"] >= start)
            & (HITTRAX_DF["DateOnly"] <= end)
        ].copy()
        if d.empty:
            return None
        d = d.sort_values("Date").reset_index(drop=True)
        return d

    @reactive.calc
    def hittrax_data_active():
        """HitTrax sessions with AB > 0 (real swings only)."""
        d = hittrax_data()
        if d is None or d.empty:
            return d
        ab = pd.to_numeric(d["AB"], errors="coerce").fillna(0)
        return d[ab > 0].reset_index(drop=True)

    def _click_to_session_key(click, d):
        """Convert a plot click's x-coordinate (data coords) to a session ISO date string."""
        if click is None or d is None or d.empty:
            return None
        x = click.get("x")
        if x is None:
            return None
        idx = int(round(float(x)))
        if idx < 0 or idx >= len(d):
            return None
        return pd.to_datetime(d["Date"].iloc[idx]).strftime("%Y-%m-%d")

    @reactive.effect
    @reactive.event(input.hittrax_max_distance_plot_click)
    def _hittrax_dist_click():
        click = input.hittrax_max_distance_plot_click()
        key = _click_to_session_key(click, hittrax_data_active())
        if key is None:
            return
        current = selected_hittrax_session.get()
        selected_hittrax_session.set("" if current == key else key)

    @reactive.effect
    @reactive.event(input.hittrax_batted_ball_mix_plot_click)
    def _hittrax_mix_click():
        click = input.hittrax_batted_ball_mix_plot_click()
        key = _click_to_session_key(click, hittrax_data_active())
        if key is None:
            return
        current = selected_hittrax_session.get()
        selected_hittrax_session.set("" if current == key else key)

    @reactive.effect
    @reactive.event(input.clicked_hittrax_session)
    def _hittrax_session_log_click():
        val = input.clicked_hittrax_session()
        if not val:
            return
        current = selected_hittrax_session.get()
        if val == "__reset__" or val == current:
            selected_hittrax_session.set("")
        else:
            selected_hittrax_session.set(str(val))

    @reactive.effect
    def _reset_hittrax_selection_on_player_change():
        """Clear session selection when player or date range changes."""
        input.player()
        input.date_start()
        input.date_end()
        input.data_source()
        selected_hittrax_session.set("")

    @reactive.calc
    def hittrax_team_benchmarks():
        """Team-level benchmarks across all Purdue HitTrax players in date range."""
        if HITTRAX_DF.empty:
            return {}
        start = input.date_start()
        end = input.date_end()
        if start is None or end is None:
            return {}

        df = HITTRAX_DF[
            (HITTRAX_DF["DateOnly"] >= start)
            & (HITTRAX_DF["DateOnly"] <= end)
        ].copy()
        # Active sessions only (AB > 0)
        ab = pd.to_numeric(df["AB"], errors="coerce").fillna(0)
        df = df[ab > 0]
        if df.empty:
            return {}

        dist = pd.to_numeric(df["Dist"], errors="coerce").dropna()
        maxv = pd.to_numeric(df["MaxV"], errors="coerce").dropna()
        avgv = pd.to_numeric(df["AvgV"], errors="coerce").dropna()

        return {
            "avg_dist": float(dist.mean()) if len(dist) else 0.0,
            "max_dist": float(dist.max()) if len(dist) else 0.0,
            "avg_max_ev": float(maxv.mean()) if len(maxv) else 0.0,
            "max_max_ev": float(maxv.max()) if len(maxv) else 0.0,
            "avg_avg_ev": float(avgv.mean()) if len(avgv) else 0.0,
            "n_sessions": int(len(df)),
            "n_players": int(df["Player"].nunique()) if "Player" in df.columns else 0,
        }

    @reactive.calc
    def hittrax_summary_text():
        """Inline summary line for the HitTrax Batter Profile header."""
        d = hittrax_data_active()
        if d is None or d.empty:
            return None
        player_raw = input.player() or ""
        name = format_display_name(player_raw) or "Batter"

        # Look up BatterSide from Trackman data (not in HitTrax files)
        side = ""
        if not MASTER_DF.empty and "Batter" in MASTER_DF.columns and "BatterSide" in MASTER_DF.columns:
            match = MASTER_DF[MASTER_DF["Batter"].astype(str).str.strip() == player_raw]
            if not match.empty:
                side_raw = str(match["BatterSide"].iloc[0]).strip()
                if side_raw and side_raw.lower() not in ("nan", ""):
                    side = side_raw

        n_sessions = len(d)
        parts = [name]
        if side:
            parts.append(f"Bats: {side}")
        parts.append(f"{n_sessions} session{'s' if n_sessions != 1 else ''}")
        return " | ".join(parts)

    @output
    @render.ui
    def hittrax_aggregate_stats():
        d = hittrax_data_active()
        if d is None or d.empty:
            return ui.div(
                "No HitTrax sessions in the selected date range.",
                style="padding:14px;color:#888;text-align:center;",
            )

        total_sessions = len(d)
        total_ab = int(pd.to_numeric(d["AB"], errors="coerce").fillna(0).sum()) if "AB" in d.columns else 0
        total_h  = int(pd.to_numeric(d["H"],  errors="coerce").fillna(0).sum()) if "H"  in d.columns else 0
        total_ebh= int(pd.to_numeric(d["EBH"],errors="coerce").fillna(0).sum()) if "EBH" in d.columns else 0
        total_hr = int(pd.to_numeric(d["HR"], errors="coerce").fillna(0).sum()) if "HR" in d.columns else 0

        # Weighted AVG / SLG by AB
        ba = (total_h / total_ab) if total_ab > 0 else 0.0
        if total_ab > 0 and "SLG" in d.columns:
            slg_series = pd.to_numeric(d["SLG"], errors="coerce").fillna(0)
            ab_series  = pd.to_numeric(d["AB"],  errors="coerce").fillna(0)
            tb = (slg_series * ab_series).sum()
            slg = tb / total_ab
        else:
            slg = 0.0
        ops = ba + slg

        # HHA weighted by AB
        def weighted_rate(col):
            if col not in d.columns or total_ab == 0:
                return 0.0
            s = pd.to_numeric(d[col], errors="coerce").fillna(0)
            a = pd.to_numeric(d["AB"], errors="coerce").fillna(0)
            return (s * a).sum() / a.sum() if a.sum() > 0 else 0.0
        hha = weighted_rate("HHA")

        # Exit velocity
        if "AvgV" in d.columns and total_ab > 0:
            avgv_series = pd.to_numeric(d["AvgV"], errors="coerce").fillna(0)
            ab_series   = pd.to_numeric(d["AB"],   errors="coerce").fillna(0)
            avg_ev = (avgv_series * ab_series).sum() / ab_series.sum() if ab_series.sum() > 0 else 0.0
        else:
            avg_ev = 0.0
        max_ev = float(pd.to_numeric(d["MaxV"], errors="coerce").max()) if "MaxV" in d.columns else 0.0

        # Distance — avg max-dist across sessions (typical) and season peak
        avg_dist = float(pd.to_numeric(d["Dist"], errors="coerce").mean()) if "Dist" in d.columns else 0.0
        max_dist = float(pd.to_numeric(d["Dist"], errors="coerce").max()) if "Dist" in d.columns else 0.0

        # Batted-ball mix weighted by AB
        ld_pct = weighted_rate("LD %")
        fb_pct = weighted_rate("FB %")
        gb_pct = weighted_rate("GB %")

        def fmt_rate(v):
            if v is None:
                return "—"
            if v >= 1.0:
                return f"{v:.3f}"
            return f".{int(round(v*1000)):03d}"

        def stat_tile(label, value, sub=""):
            return ui.div(
                ui.div(label, style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:0.5px;font-weight:700;margin-bottom:3px;"),
                ui.div(value, style="font-size:22px;font-weight:700;color:#1a3a6b;line-height:1.1;"),
                ui.div(sub, style="font-size:10px;color:#888;margin-top:1px;") if sub else None,
                style=(
                    "background:#fff;border:1px solid #e0e0e0;border-radius:8px;"
                    "padding:10px 10px;text-align:center;min-width:0;"
                ),
            )

        def category(title, accent, tiles):
            return ui.div(
                ui.div(
                    title,
                    style=(
                        f"font-size:10px;font-weight:800;color:{accent};"
                        "text-transform:uppercase;letter-spacing:1.0px;"
                        "padding:4px 0 8px 2px;"
                    ),
                ),
                ui.div(
                    *tiles,
                    style="display:grid;grid-template-columns:1fr 1fr;gap:8px;",
                ),
                style=(
                    f"padding:12px 14px;border-left:3px solid {accent};"
                    "background:#fafafa;border-radius:6px;"
                ),
            )

        return ui.div(
            ui.div(
                category("Contact", "#27ae60", [
                    stat_tile("AB", f"{total_ab}"),
                    stat_tile("H", f"{total_h}"),
                    stat_tile("AVG", fmt_rate(ba)),
                    stat_tile("HHA", fmt_rate(hha)),
                ]),
                category("Power", "#c0392b", [
                    stat_tile("EBH", f"{total_ebh}"),
                    stat_tile("HR", f"{total_hr}"),
                    stat_tile("SLG", fmt_rate(slg)),
                    stat_tile("OPS", fmt_rate(ops)),
                ]),
                category("Exit Velo", "#185FA5", [
                    stat_tile("Avg EV", f"{avg_ev:.1f}", "mph"),
                    stat_tile("Max EV", f"{max_ev:.1f}", "mph"),
                    stat_tile("Avg Dist", f"{avg_dist:.0f}", "ft"),
                    stat_tile("Max Dist", f"{max_dist:.0f}", "ft"),
                ]),
                category("Batted Ball", "#d4a017", [
                    stat_tile("LD %", f"{ld_pct:.0f}%"),
                    stat_tile("FB %", f"{fb_pct:.0f}%"),
                    stat_tile("GB %", f"{gb_pct:.0f}%"),
                    stat_tile("Sessions", f"{total_sessions}"),
                ]),
                style=(
                    "display:grid;grid-template-columns:repeat(4, 1fr);"
                    "gap:12px;padding:14px;"
                ),
            ),
        )

    @output
    @render.plot
    def hittrax_batted_ball_mix_plot():
        d = hittrax_data_active()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#fafafa")

        if d is None or d.empty:
            ax.text(0.5, 0.5, "No HitTrax data", ha="center", va="center",
                    transform=ax.transAxes, color="#888", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.close(fig)
            return fig

        dates = pd.to_datetime(d["Date"])
        ld = pd.to_numeric(d.get("LD %", pd.Series(0, index=d.index)), errors="coerce").fillna(0)
        fb = pd.to_numeric(d.get("FB %", pd.Series(0, index=d.index)), errors="coerce").fillna(0)
        gb = pd.to_numeric(d.get("GB %", pd.Series(0, index=d.index)), errors="coerce").fillna(0)

        # X positions: integer index so bars are evenly spaced
        x = np.arange(len(d))
        width = 0.7

        # Selection dimming
        sel = selected_hittrax_session.get()
        date_keys = [pd.to_datetime(dt).strftime("%Y-%m-%d") for dt in dates]
        if sel:
            alphas = [1.0 if k == sel else 0.22 for k in date_keys]
        else:
            alphas = [1.0] * len(d)

        ld_bars = ax.bar(x, ld, width, color="#27ae60", label="LD%", zorder=3)
        fb_bars = ax.bar(x, fb, width, bottom=ld, color="#3498db", label="FB%", zorder=3)
        gb_bars = ax.bar(x, gb, width, bottom=ld + fb, color="#d4a017", label="GB%", zorder=3)
        for bars_group in (ld_bars, fb_bars, gb_bars):
            for bar, a in zip(bars_group, alphas):
                bar.set_alpha(a)

        # Highlight outline on selected session
        if sel:
            for i, k in enumerate(date_keys):
                if k == sel:
                    ax.add_patch(Rectangle(
                        (x[i] - width/2, 0), width, 100,
                        fill=False, edgecolor="#1a3a6b", lw=2, zorder=4,
                    ))

        # In-bar labels (only if > 8%)
        for i, (l, f, g) in enumerate(zip(ld, fb, gb)):
            if l >= 8:
                ax.text(i, l/2, f"{int(l)}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            if f >= 8:
                ax.text(i, l + f/2, f"{int(f)}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            if g >= 8:
                ax.text(i, l + f + g/2, f"{int(g)}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([dt.strftime("%b %d") for dt in dates],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Batted Ball Mix (%)", fontsize=11, color="#444")
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(True, axis="y", color="#e0e0e0", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#ccc")
        ax.spines["bottom"].set_color("#ccc")
        ax.tick_params(colors="#666", labelsize=9)

        ax.legend(loc="upper right", frameon=True, fontsize=9,
                  framealpha=0.95, edgecolor="#ccc", ncol=3,
                  bbox_to_anchor=(1.0, 1.12))

        fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.20)
        plt.close(fig)
        return fig

    @output
    @render.plot
    def hittrax_max_distance_plot():
        d = hittrax_data_active()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#fafafa")

        if d is None or d.empty:
            ax.text(0.5, 0.5, "No HitTrax data", ha="center", va="center",
                    transform=ax.transAxes, color="#888", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.close(fig)
            return fig

        dates = pd.to_datetime(d["Date"])
        dist  = pd.to_numeric(d["Dist"], errors="coerce").fillna(0)
        maxv  = pd.to_numeric(d["MaxV"], errors="coerce").fillna(0)

        x = np.arange(len(d))
        width = 0.72

        # Color bars by Max EV — links power output to contact quality
        if (maxv > 0).any():
            ev_min = float(maxv[maxv > 0].min())
            ev_max = float(maxv.max())
        else:
            ev_min, ev_max = 0.0, 1.0
        span = max(ev_max - ev_min, 1)

        def bar_color(v):
            if v <= 0:
                return "#dddddd"
            t = (v - ev_min) / span
            t = max(0, min(1, t))
            r1, g1, b1 = 52, 152, 219
            r2, g2, b2 = 192, 57, 43
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            return f"#{r:02x}{g:02x}{b:02x}"

        colors = [bar_color(v) for v in maxv]

        # Selection: dim non-selected bars, highlight selected
        sel = selected_hittrax_session.get()
        date_keys = [pd.to_datetime(dt).strftime("%Y-%m-%d") for dt in dates]
        if sel:
            alphas = [1.0 if k == sel else 0.22 for k in date_keys]
            edge_colors = ["#1a3a6b" if k == sel else "white" for k in date_keys]
            edge_widths = [2.0 if k == sel else 0.6 for k in date_keys]
            bars = ax.bar(x, dist, width, color=colors, zorder=3,
                          edgecolor=edge_colors, lw=edge_widths)
            for bar, a in zip(bars, alphas):
                bar.set_alpha(a)
        else:
            ax.bar(x, dist, width, color=colors, zorder=3,
                   edgecolor="white", lw=0.6)

        season_max = float(dist.max()) if len(dist) else 0
        for i, v in enumerate(dist):
            if v > 0:
                ax.text(i, v + season_max * 0.02 + 2, f"{int(v)}",
                        ha="center", va="bottom",
                        fontsize=8, color="#444", fontweight="bold")

        # Team Max (red dashed) — best single-session max-dist across all Purdue HitTrax
        bench = hittrax_team_benchmarks()
        team_max = float(bench.get("max_dist", 0.0))

        if team_max > 0:
            ax.axhline(team_max, color="#c0392b", ls="--", lw=1.5, alpha=0.9, zorder=2)
            ax.text(
                len(d) - 0.5, team_max + 2,
                f"Team Max: {int(round(team_max))} ft",
                ha="right", va="bottom",
                fontsize=9, color="#c0392b", fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([dt.strftime("%b %d") for dt in dates],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Max Distance (ft)", fontsize=11, color="#444")
        # Include Team Max in y-limit so the top reference line has headroom
        y_top = max(season_max, team_max) if (season_max > 0 or team_max > 0) else 1
        ax.set_ylim(0, y_top * 1.18)
        ax.grid(True, axis="y", color="#e0e0e0", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#ccc")
        ax.spines["bottom"].set_color("#ccc")
        ax.tick_params(colors="#666", labelsize=9)

        ax.text(
            0.02, 0.95, "Bar color = session Max EV",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#666", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", lw=0.5),
        )

        fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.20)
        plt.close(fig)
        return fig

    @output
    @render.ui
    def hittrax_session_log():
        d = hittrax_data_active()
        if d is None or d.empty:
            return ui.div(
                "No sessions to display.",
                style="padding:14px;color:#888;text-align:center;",
            )

        # Sort newest first for display
        d = d.sort_values("Date", ascending=False).reset_index(drop=True)

        def fmt_rate(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return "—"
            if pd.isna(v):
                return "—"
            if v >= 1.0:
                return f"{v:.3f}"
            return f".{int(round(v*1000)):03d}"

        def fmt_int(v):
            try:
                return f"{int(float(v))}"
            except (TypeError, ValueError):
                return "—"

        def fmt_f1(v):
            try:
                v = float(v)
                if pd.isna(v):
                    return "—"
                return f"{v:.1f}"
            except (TypeError, ValueError):
                return "—"

        headers = [
            "Date", "AB", "H", "EBH", "HR",
            "AVG", "SLG", "HHA",
            "Avg EV", "Max EV", "Dist",
            "LD%", "FB%", "GB%",
        ]
        header_cells = [
            ui.tags.th(h, style=(
                "text-align:center;padding:8px 10px;font-size:11px;"
                "background:#2a3a5e;color:#fff;font-weight:700;"
                "text-transform:uppercase;letter-spacing:0.4px;"
                "border-bottom:2px solid #1a2a4e;"
            )) for h in headers
        ]

        sel = selected_hittrax_session.get()

        rows = []
        for i, r in d.iterrows():
            date_key = pd.to_datetime(r["Date"]).strftime("%Y-%m-%d")
            is_selected = (sel == date_key)

            if is_selected:
                row_bg = "#fff8e1"  # warm highlight
                font_size = "18px"
                font_weight = "800"
                padding = "12px 10px"
            else:
                row_bg = "#ffffff" if i % 2 == 0 else "#fafafa"
                font_size = "12px"
                font_weight = "400"
                padding = "7px 10px"

            border_left = "border-left:4px solid #DDB945;" if is_selected else "border-left:4px solid transparent;"
            base_td = (
                f"padding:{padding};text-align:center;font-size:{font_size};color:#222;"
                f"font-weight:{font_weight};"
                f"background:{row_bg};border-bottom:1px solid #eee;"
            )
            # Date column is always bold; selected = even bolder + bigger
            date_weight = "800" if is_selected else "700"
            date_td = (
                f"padding:{padding};font-size:{font_size};font-weight:{date_weight};color:#222;"
                f"background:{row_bg};border-bottom:1px solid #eee;white-space:nowrap;{border_left}"
            )
            cells = [
                ui.tags.td(pd.to_datetime(r["Date"]).strftime("%b %d, %Y"), style=date_td),
                ui.tags.td(fmt_int(r.get("AB", 0)),  style=base_td),
                ui.tags.td(fmt_int(r.get("H", 0)),   style=base_td),
                ui.tags.td(fmt_int(r.get("EBH", 0)), style=base_td),
                ui.tags.td(fmt_int(r.get("HR", 0)),  style=base_td),
                ui.tags.td(fmt_rate(r.get("AVG", 0)), style=base_td),
                ui.tags.td(fmt_rate(r.get("SLG", 0)), style=base_td),
                ui.tags.td(fmt_rate(r.get("HHA", 0)), style=base_td),
                ui.tags.td(fmt_f1(r.get("AvgV", 0)), style=base_td),
                ui.tags.td(fmt_f1(r.get("MaxV", 0)), style=base_td),
                ui.tags.td(fmt_int(r.get("Dist", 0)), style=base_td),
                ui.tags.td(f"{fmt_int(r.get('LD %', 0))}%",
                           style=base_td + f"color:#27ae60;font-weight:{'800' if is_selected else '700'};"),
                ui.tags.td(f"{fmt_int(r.get('FB %', 0))}%",
                           style=base_td + f"color:#3498db;font-weight:{'800' if is_selected else '700'};"),
                ui.tags.td(f"{fmt_int(r.get('GB %', 0))}%",
                           style=base_td + f"color:#d4a017;font-weight:{'800' if is_selected else '700'};"),
            ]
            rows.append(
                ui.tags.tr(
                    *cells,
                    {"data-session-key": date_key, "class": "hittrax-session-row"},
                    style="cursor:pointer;",
                )
            )

        return ui.div(
            ui.tags.table(
                ui.tags.thead(ui.tags.tr(*header_cells)),
                ui.tags.tbody(*rows),
                style="width:100%;border-collapse:collapse;",
            ),
            ui.tags.script(ui.HTML(
                "(function(){"
                "  $(document).off('click.hittraxrow').on('click.hittraxrow', '.hittrax-session-row', function(){"
                "    var k = $(this).attr('data-session-key');"
                "    if (k) { Shiny.setInputValue('clicked_hittrax_session', k, {priority:'event'}); }"
                "  });"
                "})();"
            )),
            style="padding:0;overflow-x:auto;",
        )

app = App(app_ui, server, static_assets=STATIC_DIR)
