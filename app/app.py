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

def load_and_clean_csv(full_path: str) -> pd.DataFrame | None:
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

    return df

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
    "Sosa, Gabriela",
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

    for rel, full, dmin, dmax in csv_paths_with_dates:
        df = load_and_clean_csv(full)
        if df is None or df.empty or "Date" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].notna()].copy()
        if df.empty:
            continue

        df["DateOnly"] = df["Date"].dt.date
        df = infer_session_type_for_purdue(df, filename=rel)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

MASTER_DF = build_master_df()

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


# ---------------------------------------------------------------------------
# UI (Sketch Layout + Purdue Header)
# ---------------------------------------------------------------------------
app_ui = ui.page_fluid(
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
                    
                    selected=DEFAULT_SEASON,
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

        class_="layout-main",
    )
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):
    selected_pitch = reactive.Value("")
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

        ui.update_date(
            "date_start",
            min=global_date_min,
            max=end,
            session=session,
        )
        ui.update_date(
            "date_end",
            min=start,
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

        return MASTER_DF[
            (MASTER_DF["DateOnly"] >= start) &
            (MASTER_DF["DateOnly"] <= end)
        ]

    @reactive.effect
    def _warn_unsupported_source():
        src = input.data_source()
        if src != "trackman":
            ui.notification_show(
                f"'{input.data_source()}' data is not yet available.",
                type="warning",
                duration=4,
            )
            ui.update_select("data_source", selected="trackman", session=session)

    @reactive.effect
    def _update_team_choices():
        df = current_df()
        if df is None or df.empty:
            ui.update_selectize("team", choices={}, session=session)
            return

        teams = set()
        for col in ["PitcherTeam", "BatterTeam"]:
            if col in df.columns:
                vals = df[col].dropna().astype(str).str.strip().tolist()
                teams.update([v for v in vals if v])

        teams = sorted(list(teams))
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
        if session_val == "bullpen" and ptype == "batter":
            return "⚠️  Bullpen sessions only contain pitcher data. Switch Player Type to Pitcher to view the profile."
        if session_val == "batting_practice" and ptype == "pitcher":
            return "⚠️  Batting Practice sessions only contain batter data. Switch Player Type to Batter to view the profile."
        return None


    # -----------------------------------------------------------------------
    # Comparison tab reactives
    # -----------------------------------------------------------------------
    @reactive.effect
    def _update_cmp_opponent_teams():
        df = current_df()
        if df is None or df.empty or "PitcherTeam" not in df.columns:
            ui.update_select("cmp_opponent_team", choices={}, session=session)
            return

        teams = sorted([
            t for t in df["PitcherTeam"].dropna().astype(str).str.strip().unique().tolist()
            if t 
        ])

        choices = {t: TEAM_NAME_MAP.get(t, t) for t in teams}
        choices = dict(sorted(choices.items(), key=lambda x: x[1]))

        ui.update_select(
            "cmp_opponent_team",
            choices=choices,
            selected=list(choices.keys())[0] if choices else None,
            session=session,
        )

        
    @reactive.effect
    def _update_cmp_opponent_pitchers():
        df = current_df()
        opp_team = input.cmp_opponent_team()

        if df is None or df.empty or not opp_team:
            ui.update_select("cmp_opponent_pitcher", choices={}, session=session)
            return

        d = df[df["PitcherTeam"].astype(str).str.strip() == str(opp_team)]

        if d.empty or "PitcherId" not in d.columns or "Pitcher" not in d.columns:
            ui.update_select("cmp_opponent_pitcher", choices={}, session=session)
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

        ui.update_select(
            "cmp_opponent_pitcher",
            choices=choices,
            selected=list(choices.keys())[0] if choices else None,
            session=session,
        )


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
                full_html=False, include_plotlyjs="cdn",
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
            full_html=False, include_plotlyjs="cdn",
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
                            ui.input_select("cmp_opponent_pitcher", "Opponent Pitcher", choices={}),
                            ui.input_select(
                                "cmp_pitch_type",
                                "Pitch Type Comparison",
                                choices={"all": "All Pitches"},
                                selected="all",
                            ),
                            class_="cmp-filter-row",
                        ),
                        ui.output_ui("cmp_team_summary_table"),
                        ui.output_ui("cmp_summary_cards"),
                        ui.div(
                            ui.div("Location by Count", class_="team-summary-title"),
                            ui.output_plot("cmp_count_location", height = "2600px"),
                            class_="team-summary-wrap",
                        ),
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
            PLOT_H = "340px"
            SPRAY_H = "260px"
            LOC_H = "670px"

            return ui.TagList(
                ui.div(
                    # Batting Summary (total row)
                    ui.div(
                        ui.div("Batting Summary", style=(
                            "font-size:13px;font-weight:700;color:#444;"
                            "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                            "background:#f3f3f3;border-radius:10px 10px 0 0;"
                        )),
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
                    ui.div(
                        # Left column: Location
                        ui.div(
                            ui.div(
                                ui.div("Pitch Location by Result", style=(
                                    "font-size:13px;font-weight:700;color:#444;"
                                    "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                                    "background:#f3f3f3;border-radius:10px 10px 0 0;"
                                )),
                                ui.output_plot("batter_location_plot", height=LOC_H),
                                class_="card",
                            ),
                        ),
                        # Right column: Spray + EV/LA + Radar/Dist
                        ui.div(
                            ui.div(
                                ui.div("Spray Chart", style=(
                                    "font-size:13px;font-weight:700;color:#444;"
                                    "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                                    "background:#f3f3f3;border-radius:10px 10px 0 0;"
                                )),
                                ui.output_plot("batter_spray_plot", height=SPRAY_H),
                                class_="card",
                            ),
                            ui.div(
                                ui.div(
                                    ui.div(
                                        ui.div("Exit Velo vs. Launch Angle", style=(
                                            "font-size:13px;font-weight:700;color:#444;"
                                            "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                                            "background:#f3f3f3;border-radius:10px 10px 0 0;"
                                        )),
                                        ui.output_plot("batter_ev_la_plot", height=PLOT_H),
                                        class_="card",
                                    ),
                                    ui.div(
                                        ui.div(bot_title, style=(
                                            "font-size:13px;font-weight:700;color:#444;"
                                            "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                                            "background:#f3f3f3;border-radius:10px 10px 0 0;"
                                        )),
                                        ui.output_plot(bot_plot, height=PLOT_H),
                                        class_="card",
                                    ),
                                    style="display:grid;grid-template-columns:1fr 1fr;gap:14px;",
                                ),
                            ),
                            style="display:flex;flex-direction:column;gap:14px;",
                        ),
                        style="display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start;",
                    ),
                    # Scout Insights (full-width, below charts)
                    ui.div(
                        ui.div("Scout Insights", style=(
                            "font-size:13px;font-weight:700;color:#444;"
                            "padding:8px 12px;border-bottom:1px solid #e0e0e0;"
                            "background:#f3f3f3;border-radius:10px 10px 0 0;"
                        )),
                        ui.output_ui("batter_scout_insights"),
                        class_="card",
                        style="margin-top:14px;",
                    ),
                ),
            )



        # Pitcher profile
        data = pitcher_data()
        if data is None or data.empty:
            return ui.div("No pitcher data for the selected filters.")
        return ui.div(
            ui.row(
                ui.column(4, ui.card(ui.card_header("Pitch Usage"),     ui.output_plot("pie",      height="340px"))),
                ui.column(4, ui.card(ui.card_header("Pitch Locations"), ui.output_plot("location", height="340px"))),
                ui.column(4, ui.card(ui.card_header("Pitch Movements"), ui.output_ui("movement"))),
            ),
            ui.row(
                ui.column(12, ui.card(
                    ui.card_header("Summary Table"),
                    ui.div(ui.output_table("usage_table"), class_="usage-table-wrap"),
                )),
            ),
        )

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
    @render.plot
    def location():
        loc = pitcher_loc_data()
        colors = pitch_colors()

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#f7f7f7")

        if loc is None or loc.empty:
            ax.text(0.5, 0.5, "No pitch location data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            plt.close(fig)
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

        sel = selected_pitch.get()

        for pt, g in loc.groupby(PITCH_TYPE_COL):
            x = g["PlateLocSide"]
            y = g["PlateLocHeight"]
            color = colors.get(pt, (0.5, 0.5, 0.5))

            if sel and pt == sel:
                ax.scatter(
                    x,
                    y,
                    s=130,
                    alpha=0.18,
                    color=color,
                    linewidths=0,
                    zorder=1,
                )

            ax.scatter(
                x,
                y,
                s=70 if sel and pt == sel else 35,
                alpha=1.0 if (not sel or pt == sel) else 0.18,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=3 if sel and pt == sel else 2,
            )

        ax.add_patch(home_plate_polygon(y_front=0.10))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PlateLocSide")
        ax.set_ylabel("PlateLocHeight")
        ax.grid(True, alpha=0.2)

        plt.close(fig)
        return fig


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

            # Selected row style (bigger + bold)
            selected_style_main = (
                "font-size:18px;"
                "font-weight:900;"
                "color:#000000;"
                "border-left:5px solid #DDB945;"
            )

            selected_style_other = (
                "font-size:16px;"
                "font-weight:800;"
                "color:#111111;"
                "background-color:#fff8dc;"
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


    # ── dynamic home tab header ──────────────────────────────────────────────
    @output
    @render.ui
    def home_profile_header():
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
            return "—" if v is None else f".{int(round(v * 1000)):03d}"

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
        return (
            out.style
            .hide(axis="index")
            .apply(lambda col: ["font-weight:700;font-size:13px;color:#222;" for _ in col], axis=0)
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
            return "—" if v is None else f".{int(round(v * 1000)):03d}"
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
            selected_style_main = (
                "font-size:18px;font-weight:900;color:#000000;"
                "border-left:5px solid #DDB945;"
            )
            selected_style_other = (
                "font-size:16px;font-weight:800;color:#111111;"
                "background-color:#fff8dc;"
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

    # ── pitch location heatmap ───────────────────────────────────────────────
    @output
    @render.plot
    def batter_location_plot():
        data = batter_data()
        bid = input.player() if input.player_type() == "batter" else None

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#f7f7f7")
        ax.set_facecolor("#f7f7f7")
        ax.set_box_aspect(1)

        SWING_EVENTS = {
            "StrikeSwinging",
            "FoulBallFieldable",
            "FoulBallNotFieldable",
            "InPlay",
        }
        CONTACT_EVENTS = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        # Shape + color for outside-zone dots (distinct from pitch type colors)
        DOT_STYLE = {
            "StrikeCalled":        {"marker": "o", "color": "#999999"},  # circle = take
            "Ball":                {"marker": "o", "color": "#999999"},
            "BallCalled":          {"marker": "o", "color": "#999999"},
            "StrikeSwinging":      {"marker": "X", "color": "#E24B4A"},  # X = swing & miss
            "FoulBallFieldable":   {"marker": "^", "color": "#7B68AE"},  # triangle = foul
            "FoulBallNotFieldable":{"marker": "^", "color": "#7B68AE"},
            "InPlay":              {"marker": "D", "color": "#2E8B57"},  # diamond = in play
        }

        zw = (ZONE_RIGHT - ZONE_LEFT) / 3
        zh = (ZONE_TOP - ZONE_BOTTOM) / 3
        col_edges = [ZONE_LEFT + i * zw for i in range(4)]
        row_edges = [ZONE_BOTTOM + i * zh for i in range(4)]

        pitcher_hand = ""
        total_pitches = 0
        total_swings = 0
        total_contact = 0

        if data is not None and not data.empty and bid:
            d = data[data["BatterId"].astype(str) == str(bid)].copy()

            # pitch type filter
            sel = batter_selected_pitch()
            if sel and PITCH_TYPE_COL in d.columns:
                d = d[d[PITCH_TYPE_COL].astype(str).str.strip() == sel]

            for c in ["PlateLocSide", "PlateLocHeight"]:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"])

            if "PitcherThrows" in d.columns:
                pt = d["PitcherThrows"].dropna().astype(str).str.strip()
                if not pt.empty:
                    val = pt.iloc[0].lower()
                    pitcher_hand = "RHP" if "right" in val else "LHP" if "left" in val else ""

            if "PitchCall" not in d.columns:
                d["PitchCall"] = ""
            pc = d["PitchCall"].astype(str).str.strip()

            total_pitches = len(d)
            total_swings = int(pc.isin(SWING_EVENTS).sum())
            total_contact = int(pc.isin(CONTACT_EVENTS).sum())

            d["is_in_zone"] = (
                d["PlateLocSide"].between(ZONE_LEFT, ZONE_RIGHT)
                & d["PlateLocHeight"].between(ZONE_BOTTOM, ZONE_TOP)
            )

            shadow_pad_x = 0.35
            shadow_pad_y = 0.30
            ax.add_patch(Rectangle(
                (ZONE_LEFT - shadow_pad_x, ZONE_BOTTOM - shadow_pad_y),
                (ZONE_RIGHT - ZONE_LEFT) + 2 * shadow_pad_x,
                (ZONE_TOP - ZONE_BOTTOM) + 2 * shadow_pad_y,
                facecolor="#edf4ee",
                edgecolor="none",
                zorder=0,
            ))

            for row_i in range(3):
                for col_i in range(3):
                    x0 = col_edges[col_i]
                    x1 = col_edges[col_i + 1]
                    y0 = row_edges[row_i]
                    y1 = row_edges[row_i + 1]
                    cx = (x0 + x1) / 2
                    cy = (y0 + y1) / 2

                    mask = (
                        d["PlateLocSide"].between(x0, x1)
                        & d["PlateLocHeight"].between(y0, y1)
                    )
                    cell_d = d[mask]
                    n_cell = len(cell_d)
                    n_swing = int(pc[mask].isin(SWING_EVENTS).sum())

                    if n_cell == 0:
                        cell_bg = "#f5f5f5"
                    else:
                        swing_pct = n_swing / n_cell
                        if swing_pct >= 0.75:
                            cell_bg = "#c0392b"
                        elif swing_pct >= 0.50:
                            cell_bg = "#f0b000"
                        elif swing_pct >= 0.25:
                            cell_bg = "#f5dfb3"
                        else:
                            cell_bg = "#ffffff"

                    ax.add_patch(Rectangle(
                        (x0, y0), zw, zh,
                        facecolor=cell_bg,
                        edgecolor="#999999",
                        linewidth=1.2,
                        zorder=2,
                    ))

                    if n_cell == 0:
                        ax.text(
                            cx, cy, "—",
                            ha="center", va="center",
                            fontsize=13, color="#aaaaaa", zorder=3
                        )
                        continue

                    swing_pct = n_swing / n_cell
                    pct_txt = f"{int(round(swing_pct * 100))}%"
                    frac_txt = f"{n_swing}/{n_cell}"

                    txt_color = "#ffffff" if swing_pct >= 0.50 else "#222222"
                    sub_color = "#ffe0db" if swing_pct >= 0.75 else "#666666"

                    ax.text(
                        cx, cy + zh * 0.16, pct_txt,
                        ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color=txt_color, zorder=3,
                    )
                    ax.text(
                        cx, cy - zh * 0.10, frac_txt,
                        ha="center", va="center",
                        fontsize=10,
                        color=sub_color if swing_pct >= 0.50 else "#666666",
                        zorder=3,
                    )

            outside = d[~d["is_in_zone"]].copy()
            if not outside.empty:
                outside["_call"] = outside["PitchCall"].astype(str).str.strip()
                for call_val, grp in outside.groupby("_call", sort=False):
                    sty = DOT_STYLE.get(call_val, {"marker": "o", "color": "#cccccc"})
                    ax.scatter(
                        grp["PlateLocSide"], grp["PlateLocHeight"],
                        s=70, marker=sty["marker"], color=sty["color"], alpha=0.85,
                        edgecolors="white", linewidth=0.5, zorder=4,
                    )

            ax.add_patch(Rectangle(
                (ZONE_LEFT, ZONE_BOTTOM),
                ZONE_RIGHT - ZONE_LEFT,
                ZONE_TOP - ZONE_BOTTOM,
                fill=False,
                linewidth=2.5,
                edgecolor="#222222",
                zorder=5,
            ))

        else:
            for row_i in range(3):
                for col_i in range(3):
                    x0 = col_edges[col_i]
                    y0 = row_edges[row_i]
                    ax.add_patch(Rectangle(
                        (x0, y0), zw, zh,
                        facecolor="white",
                        edgecolor="#aaaaaa",
                        linewidth=1.2,
                        zorder=2,
                    ))
            ax.add_patch(Rectangle(
                (ZONE_LEFT, ZONE_BOTTOM),
                ZONE_RIGHT - ZONE_LEFT,
                ZONE_TOP - ZONE_BOTTOM,
                fill=False,
                linewidth=2.5,
                edgecolor="#222222",
                zorder=5,
            ))

        s_pct = f"{int(round(total_swings / total_pitches * 100))}%" if total_pitches > 0 else "—"
        c_pct = f"{int(round(total_contact / total_swings * 100))}%" if total_swings > 0 else "—"

        vs_str = f" | vs {pitcher_hand}" if pitcher_hand else ""
        subtitle = f"Swing%: {s_pct}   |   Contact%: {c_pct}   |   n = {total_pitches}{vs_str}"

        ax.text(
            0.5, 1.03, subtitle,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=10, color="#555555",
        )

        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#999999", markersize=9, label="Take"),
            plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#E24B4A", markeredgecolor="#E24B4A", markersize=9, label="Swing & Miss"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#7B68AE", markersize=9, label="Foul"),
            plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#2E8B57", markersize=9, label="In Play"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=4,
            frameon=True,
            fontsize=9,
            handletextpad=0.4,
            columnspacing=1.2,
        )

        ax.set_xlim(ZONE_LEFT - 1.0, ZONE_RIGHT + 1.0)
        ax.set_ylim(ZONE_BOTTOM - 0.6, ZONE_TOP + 0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.subplots_adjust(top=0.88, bottom=0.20, left=0.08, right=0.92)
        
        plt.close(fig)
        return fig

    # ── scout insights ──────────────────────────────────────────────────────
    @output
    @render.ui
    def batter_scout_insights():
        data = batter_data()
        bid = input.player() if input.player_type() == "batter" else None
        if data is None or data.empty or not bid:
            return ui.div()

        d = data[data["BatterId"].astype(str) == str(bid)].copy()
        sel = batter_selected_pitch()
        if sel and PITCH_TYPE_COL in d.columns:
            d = d[d[PITCH_TYPE_COL].astype(str).str.strip() == sel]

        if d.empty:
            return ui.div()

        for c in ["PlateLocSide", "PlateLocHeight"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"])

        if d.empty:
            return ui.div()

        SWING_EV = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        CONTACT_EV = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        calls = d["PitchCall"].astype(str).str.strip() if "PitchCall" in d.columns else pd.Series("", index=d.index)

        zw = (ZONE_RIGHT - ZONE_LEFT) / 3
        zh = (ZONE_TOP - ZONE_BOTTOM) / 3
        ROW_LABELS = ["lower", "middle", "upper"]
        COL_LABELS = ["inside", "middle", "outside"]
        MIN_PITCHES = 4

        # Collect candidates per type — keep only the strongest
        best = {}  # type -> (score, label, color, desc)

        # In-zone 3x3 cells
        for ri in range(3):
            for ci in range(3):
                left = ZONE_LEFT + ci * zw
                right = left + zw
                bot = ZONE_BOTTOM + ri * zh
                top = bot + zh
                mask = (
                    (d["PlateLocSide"] >= left) & (d["PlateLocSide"] < right) &
                    (d["PlateLocHeight"] >= bot) & (d["PlateLocHeight"] < top)
                )
                cell = d[mask]
                n = len(cell)
                if n < MIN_PITCHES:
                    continue
                cell_calls = calls[mask]
                n_swing = cell_calls.isin(SWING_EV).sum()
                n_contact = cell_calls.isin(CONTACT_EV).sum()
                sw_pct = n_swing / n if n > 0 else 0
                ct_pct = n_contact / n_swing if n_swing > 0 else 0
                loc = f"{ROW_LABELS[ri]}-{COL_LABELS[ci]}"

                if sw_pct >= 0.70 and ct_pct >= 0.80:
                    score = sw_pct * ct_pct * n
                    if "Hot Zone" not in best or score > best["Hot Zone"][0]:
                        best["Hot Zone"] = (score, "ATTACK ZONE", "#c0392b", loc,
                            f"{sw_pct:.0%} swing rate, {ct_pct:.0%} contact. "
                            "Aggressive and making hard contact. Do not pitch here.")

                if sw_pct <= 0.30:
                    score = (1 - sw_pct) * n
                    if "Disciplined" not in best or score > best["Disciplined"][0]:
                        best["Disciplined"] = (score, "SAFE ZONE", "#1D9E75", loc,
                            f"Only {sw_pct:.0%} swing rate on {n} pitches. "
                            "Very passive in this area. Live here early in counts to steal called strikes.")

                if sw_pct >= 0.50 and ct_pct < 0.50 and n_swing >= MIN_PITCHES:
                    score = sw_pct * (1 - ct_pct) * n
                    if "Miss Spot" not in best or score > best["Miss Spot"][0]:
                        best["Miss Spot"] = (score, "STRIKEOUT ZONE", "#7B2FBE", loc,
                            f"{sw_pct:.0%} swing rate, only {ct_pct:.0%} contact. "
                            "Will commit but can't catch up. Go-to spot for the punchout.")

        # Out-of-zone regions
        PAD_H, PAD_V = 0.5, 0.5
        out_regions = [
            ("above the zone", ZONE_LEFT, ZONE_RIGHT, ZONE_TOP, ZONE_TOP + PAD_V),
            ("below the zone", ZONE_LEFT, ZONE_RIGHT, ZONE_BOTTOM - PAD_V, ZONE_BOTTOM),
            ("inside",  ZONE_LEFT - PAD_H, ZONE_LEFT, ZONE_BOTTOM, ZONE_TOP),
            ("outside", ZONE_RIGHT, ZONE_RIGHT + PAD_H, ZONE_BOTTOM, ZONE_TOP),
        ]
        for label, ol, orr, ob, ot in out_regions:
            mask = (
                (d["PlateLocSide"] >= ol) & (d["PlateLocSide"] < orr) &
                (d["PlateLocHeight"] >= ob) & (d["PlateLocHeight"] < ot)
            )
            cell = d[mask]
            n = len(cell)
            if n < MIN_PITCHES:
                continue
            cell_calls = calls[mask]
            n_swing = cell_calls.isin(SWING_EV).sum()
            chase_pct = n_swing / n if n > 0 else 0
            if chase_pct >= 0.50:
                score = chase_pct * n
                if "Chases" not in best or score > best["Chases"][0]:
                    best["Chases"] = (score, "EXPAND HERE", "#E67E22", label,
                        f"{chase_pct:.0%} chase rate on {n} pitches. "
                        "Tunnel something that looks like a strike and let it run off the plate.")
            elif chase_pct <= 0.15:
                score = (1 - chase_pct) * n
                if "Patient" not in best or score > best["Patient"][0]:
                    best["Patient"] = (score, "DON'T WASTE", "#2980B9", label,
                        f"Only {chase_pct:.0%} chase rate on {n} pitches. "
                        "Not expanding here. Compete with strikes, don't waste pitches.")

        # ── Batted ball insights (spray, EV/LA) ──
        d_full = data[data["BatterId"].astype(str) == str(bid)].copy()
        if sel and PITCH_TYPE_COL in d_full.columns:
            d_full = d_full[d_full[PITCH_TYPE_COL].astype(str).str.strip() == sel]

        if "ExitSpeed" in d_full.columns and "Angle" in d_full.columns:
            ev = pd.to_numeric(d_full["ExitSpeed"], errors="coerce")
            la = pd.to_numeric(d_full["Angle"], errors="coerce")
            valid_bb = d_full[ev.notna() & la.notna() & (ev > 0)].copy()
            valid_bb["_ev"] = pd.to_numeric(valid_bb["ExitSpeed"], errors="coerce")
            valid_bb["_la"] = pd.to_numeric(valid_bb["Angle"], errors="coerce")

            if len(valid_bb) >= 5:
                avg_ev = valid_bb["_ev"].mean()
                barrel_count = ((valid_bb["_ev"] >= 98) & (valid_bb["_la"].between(26, 30))).sum()
                hard_hit_pct = (valid_bb["_ev"] >= 80).mean()

                if hard_hit_pct >= 0.40:
                    best["Hard Hit"] = (hard_hit_pct, "HARD CONTACT", "#8B0000", "batted balls",
                        f"{hard_hit_pct:.0%} hard-hit rate (80+ mph), avg exit velo {avg_ev:.1f} mph. "
                        "Dangerous hitter. Keep the ball off the barrel, work edges and change speeds.")
                elif hard_hit_pct <= 0.15 and len(valid_bb) >= 8:
                    best["Weak Contact"] = (1 - hard_hit_pct, "WEAK CONTACT", "#6B8E23", "batted balls",
                        f"Only {hard_hit_pct:.0%} hard-hit rate, avg exit velo {avg_ev:.1f} mph. "
                        "Not generating power. Attack the zone and let the defense work.")

                if barrel_count >= 3:
                    best["Barrel"] = (barrel_count, "BARREL THREAT", "#8B0000", "batted balls",
                        f"{barrel_count} barrels (98+ mph, 26-30 launch angle) on {len(valid_bb)} batted balls. "
                        "Can do damage when squared up. Avoid pitches in the happy zone.")

        # ── Spray tendency insights ──
        if "Direction" in d_full.columns and "PlayResult" in d_full.columns:
            pr = d_full["PlayResult"].astype(str).str.strip()
            hits = d_full[pr.isin({"Single", "Double", "Triple", "HomeRun"})].copy()
            if len(hits) >= 5:
                dirs = pd.to_numeric(hits["Direction"], errors="coerce").dropna()
                if len(dirs) >= 5:
                    pull_pct = (dirs < -15).mean()
                    oppo_pct = (dirs > 15).mean()
                    if pull_pct >= 0.60:
                        best["Pull"] = (pull_pct, "PULL HEAVY", "#8B4513", "spray chart",
                            f"{pull_pct:.0%} of hits go to the pull side. "
                            "Shade the defense pull-side. Pitch away to neutralize.")
                    elif oppo_pct >= 0.50:
                        best["Oppo"] = (oppo_pct, "USES ALL FIELDS", "#2E8B57", "spray chart",
                            f"{oppo_pct:.0%} of hits go opposite field. "
                            "Can't rely on pull-side shifts. Must locate to both sides of the plate.")

        # ── Plate discipline insights (radar metrics) ──
        if "PitchCall" in d_full.columns:
            pc_all = d_full["PitchCall"].astype(str).str.strip()
            SW_ALL = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
            total_p = len(d_full)
            total_sw = pc_all.isin(SW_ALL).sum()
            total_whiff = (pc_all == "StrikeSwinging").sum()

            if total_sw >= 10:
                whiff_rate = total_whiff / total_sw
                if whiff_rate >= 0.25:
                    best["Whiff"] = (whiff_rate, "HIGH WHIFF", "#7B2FBE", "plate discipline",
                        f"{whiff_rate:.0%} whiff rate across all swings. "
                        "Swing-and-miss in the arsenal. Can be put away with two strikes.")
                elif whiff_rate <= 0.08 and total_sw >= 20:
                    best["Contact"] = (1 - whiff_rate, "BAT-TO-BALL", "#2E8B57", "plate discipline",
                        f"Only {whiff_rate:.0%} whiff rate on {total_sw} swings. "
                        "Rarely misses. Must locate and induce weak contact, strikeouts will be tough.")

        if not best:
            return ui.div(
                ui.tags.span("Insufficient data for scouting insights.",
                             style="font-size:12px;color:#888;font-style:italic;"),
                style="padding:12px 16px;",
            )

        # Order by priority
        TYPE_ORDER = [
            "Hot Zone", "Miss Spot", "Chases", "Disciplined", "Patient",
            "Hard Hit", "Weak Contact", "Barrel",
            "Pull", "Oppo",
            "Whiff", "Contact",
        ]
        ordered = [best[t] for t in TYPE_ORDER if t in best]
        ordered = ordered[:7]  # cap at 7 insights

        items = []
        for _, badge, color, loc, desc in ordered:
            items.append(ui.tags.div(
                ui.tags.span(
                    badge,
                    style=(
                        f"display:inline-block;background:{color};color:#fff;"
                        "font-size:9px;font-weight:800;letter-spacing:0.5px;"
                        "padding:2px 8px;border-radius:3px;margin-right:8px;"
                        "white-space:nowrap;"
                    ),
                ),
                ui.tags.span(
                    f"{loc}: ",
                    style="font-weight:700;font-size:12px;color:#222;margin-right:2px;",
                ),
                ui.tags.span(
                    desc,
                    style="font-size:12px;line-height:1.5;color:#444;",
                ),
                style=(
                    "display:flex;align-items:center;flex-wrap:wrap;padding:8px 0;"
                    "border-bottom:1px solid #f0f0f0;"
                ),
            ))

        return ui.div(*items, style="padding:12px 16px;")

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
        ax.legend(handles=[
            plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=c,markersize=7,label=l)
            for l,c in [("Single","#1D9E75"),("Double","#378ADD"),
                        ("Triple","#7B2FBE"),("HR","#BA7517"),("Out","#D85A30")]
        ], loc="lower center", ncol=5, fontsize=7, framealpha=0.8)
        fig.tight_layout()

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
        ax.set_xlim(40, 120)
        ax.set_ylim(-20, 70)
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

        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, label=l)
                for l, c in [("GB", "#D85A30"), ("LD", "#1D9E75"), ("FB", "#378ADD"), ("Popup", "#888780")]
            ],
            loc="upper left",
            fontsize=7,
            framealpha=0.85,
            ncol=2,
        )

        fig.subplots_adjust(top=0.96, bottom=0.14, left=0.14, right=0.96)

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

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
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
                fontsize=7.5, fontweight="bold",
                color="#185FA5", zorder=5,
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.50, 0.75])
        ax.set_yticklabels(["25%", "50%", "75%"], fontsize=7, color="#aaa")
        ax.spines["polar"].set_visible(False)

        fig.subplots_adjust(top=0.96, bottom=0.10, left=0.08, right=0.92)

        plt.close(fig)
        return fig


app = App(app_ui, server, static_assets=STATIC_DIR)
