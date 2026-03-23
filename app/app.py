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
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch

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
    if batter_has_purdue and pitcher_has_purdue and pitcher_has_data and batter_is_active:
        out["SessionType"] = "scrimmage"
        out["SessionTypeReason"] = "Purdue vs Purdue, active roster"
        return out

    # 5. Unknown
    out["SessionType"] = "unknown"
    out["SessionTypeReason"] = "Could not classify"
    return out


def apply_session_filter_for_team(df, team, session_value):
    if df is None or df.empty:
        return df

    out = df.copy()

    if session_value != "all":
        out = out[out["SessionType"] == session_value]

    return out


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


def is_valid_pitch_type(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return (
        series.notna()
        & s.ne("")
        & s.ne("undefined")
        & s.ne("other")
        & s.ne("nan")
    )


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
            width: 160px;           /* reserves space so center title stays centered */
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
            width: 160px;           /* matches logo-wrap width for perfect centering */
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
            overflow: auto;
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

        /* Usage table styling */
        .usage-table-wrap table {
            width: 100% !important;
            table-layout: auto !important;
            border-collapse: collapse !important;
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

            ui.input_select("team", "Team Name", choices={}),
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
        if start is None or end is None or not csv_paths_with_dates:
            return None
        if start > end:
            return None

        dfs = []
        for rel, full, dmin, dmax in csv_paths_with_dates:
            if dmax < start or dmin > end:
                continue

            df = load_and_clean_csv(full)
            if df is None or "Date" not in df.columns:
                continue

            df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
            df = df[df["_date"].notna() & (df["_date"] >= start) & (df["_date"] <= end)]
            df = df.drop(columns=["_date"], errors="ignore")

            if df.empty:
                continue

            df = infer_session_type_for_purdue(df, filename=rel)
            dfs.append(df)

        if not dfs:
            return None

        combined = pd.concat(dfs, ignore_index=True)
        return combined

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
            ui.update_select("team", choices={}, session=session)
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

        d = df[df["PitcherTeam"].astype(str).str.strip() == str(opp_team)].copy()

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
    @render.plot
    def cmp_movement():
        pur = cmp_purdue_filtered()
        opp = cmp_opponent_filtered()

        fig, ax = plt.subplots(figsize=(10.5, 4.2))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        if pur is None or pur.empty or opp is None or opp.empty:
            ax.text(0.5, 0.5, "No comparison movement data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        pur_mov = pur[pur[X_MOV].notna() & pur[Y_MOV].notna()].copy()
        opp_mov = opp[opp[X_MOV].notna() & opp[Y_MOV].notna()].copy()

        if pur_mov.empty and opp_mov.empty:
            ax.text(0.5, 0.5, "No comparison movement data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        if not pur_mov.empty:
            ax.scatter(
                pur_mov[X_MOV], pur_mov[Y_MOV],
                s=18, alpha=0.75, color="#DDB945", label="Purdue"
            )

        if not opp_mov.empty:
            ax.scatter(
                opp_mov[X_MOV], opp_mov[Y_MOV],
                s=18, alpha=0.55, color="#9E9E9E", label="Opponent"
            )

        ax.axhline(0, linewidth=1, color="#777777")
        ax.axvline(0, linewidth=1, color="#777777")
        # set axis range
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect("equal", adjustable = "box")
        ax.set_xlabel("Horizontal Break (in)")
        ax.set_ylabel("Induced Vertical Break (in)")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02,1), borderaxespad=0)
        ax.set_title(cmp_pitch_type_label(), fontsize=13, fontweight="bold", loc ="center")
        return fig

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
            ].copy()

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
            return d.loc[mask].copy()

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
                loc = df[df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()].copy()
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
                        ui.div("Pitch Profile", class_="profile-title"),
                        ui.div(ui.output_text("player_summary"), class_="player-summary"),
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

                ui.nav_panel(
                    "Development",
                    ui.div(
                        ui.div("Development", class_="profile-title"),
                        ui.input_select(
                            "dev_view",
                            "",
                            choices={"strike_whiff": "Strike & Whiff Trends"},
                            selected="strike_whiff",
                        ),
                        ui.br(),
                        ui.output_plot("dev_strike_whiff_trend", height="800px"),
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
            return ui.div(
                warning,
                style=(
                    "padding:14px 18px; border-radius:8px; background:#fef9ec;"
                    "border:1.5px solid #DDB945; color:#6b4e00; "
                    "font-size:15px; font-weight:700; margin-top:12px;"
                ),
            )

        if input.player_type() != "pitcher":
            return ui.div("Batter view will be added next. Switch Player Type to Pitcher for now.")

        data = pitcher_data()
        if data is None or data.empty:
            return ui.div("No pitcher data for the selected filters. Try another pitcher or date range.")

        return ui.div(
            ui.row(
                ui.column(
                    4,
                    ui.card(
                        ui.card_header("Pitch Usage"),
                        ui.output_plot("pie", height="340px"),
                    ),
                ),
                ui.column(
                    4,
                    ui.card(
                        ui.card_header("Pitch Locations"),
                        ui.output_plot("location", height="340px"),
                    ),
                ),
                ui.column(
                    4,
                    ui.card(
                        ui.card_header("Pitch Movements"),
                        ui.output_plot("movement", height="340px"),
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
            & is_valid_pitch_type(data[PITCH_TYPE_COL])
        ].copy()
        
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

        mov = data[data[X_MOV].notna()& data[Y_MOV].notna()& is_valid_pitch_type(data[PITCH_TYPE_COL])].copy()
        if mov.empty:
            ax.text(0.5, 0.5, "No pitch movement data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        handles = []
        legend_labels = []

        for pt, g in mov.groupby(PITCH_TYPE_COL):
            sc = ax.scatter(
                g[X_MOV], g[Y_MOV],
                s=25, alpha=0.75,
                color=colors.get(pt, (0.5, 0.5, 0.5))
            )
            handles.append(sc)
            legend_labels.append(pt)

        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)
        ax.set_xlim(*MOV_XLIM)
        ax.set_ylim(*MOV_YLIM)
        ax.set_xlabel("Horizontal break (in)")
        ax.set_ylabel("Induced vertical break (in)")
        ax.grid(True, alpha=0.25)
        
        # Build-in stats box text
        lines = []
        for pt, g in mov.groupby(PITCH_TYPE_COL):
            avg_hb  = g[X_MOV].mean()
            avg_ivb = g[Y_MOV].mean()
            color   = colors.get(pt, "#555555")
            lines.append((pt, avg_hb, avg_ivb, color))

        box_text = "\n".join(
            f"{pt}: HB {avg_hb:+.1f}, IVB {avg_ivb:+.1f}"
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

    @output
    @render.ui
    def movement_legend():
        df = pitcher_data()
        if df is None or df.empty or PITCH_TYPE_COL not in df.columns:
            return ui.div()

        order = (
            df.loc[is_valid_pitch_type(df[PITCH_TYPE_COL]), PITCH_TYPE_COL]
            .astype(str)
            .value_counts()
            .index.tolist()
        )

        cols = pitch_colors()

        items = []
        for pt in order:
            if pt.lower() in {"other", "undefined"}:
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
            class_="legend-row"
        )

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
        df = df[is_valid_pitch_type(df[PITCH_TYPE_COL])].copy()

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

        # Aggregate DAILY by pitch type
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

        # Optional: remove tiny-sample days (prevents weird spikes)
        g = g[g["pitch_n"] >= 8].copy()

        # Optional: hide "Other" because it ruins readability
        g = g[g[PITCH_TYPE_COL].astype(str).str.lower() != "other"].copy()

        if g.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "Not enough pitch samples to plot trends.", ha="center", va="center")
            ax.set_axis_off()
            return fig
        xmin = g["Date"].min()
        xmax = g["Date"].max()

        # Optional smoothing (recommended). If you truly want raw, set window = 1.
        window = 1
        g["strike_plot"] = g.groupby(PITCH_TYPE_COL)["strike_pct"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        g["whiff_plot"] = g.groupby(PITCH_TYPE_COL)["whiff_pct"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

        pitch_types = list(g[PITCH_TYPE_COL].dropna().unique())
        colors = pitch_colors()

        # ---- Full-pane layout: 2-column grid ----
        n = len(pitch_types)
        ncols = 2 if n > 1 else 1
        nrows = math.ceil(n / ncols)

        # Big figure to use the whole main pane
        fig_w = 14
        fig_h = 5 * nrows   # grows with number of rows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
        fig.subplots_adjust(top=0.90, hspace=0.5, wspace=0.3)
        

        # Normalize axes to a flat list
        if nrows == 1 and ncols == 1:
            axes_list = [axes]
        elif nrows == 1:
            axes_list = list(axes)
        else:
            axes_list = [ax for row in axes for ax in (row if isinstance(row, (list, np.ndarray)) else [row])]

        fig.patch.set_facecolor("#ffffff")

        for i, pt in enumerate(pitch_types):
            ax = axes_list[i]
            sub = g[g[PITCH_TYPE_COL] == pt]

            c = colors.get(pt, "#1F3A5F")
            n_total = int(sub["pitch_n"].sum())

            ax.plot(
                sub["Date"], sub["strike_plot"] * 100,
                linewidth=2.5, marker="o", markersize=7,
                color="#DDB945", label="Strike %", zorder=3
            )
            ax.plot(
                sub["Date"], sub["whiff_plot"] * 100,
                linewidth=2.5, linestyle="--", marker="s", markersize=6,
                color="#AAAAAA", label="Whiff %", zorder=3
            )

            ax.set_xlim(sub["Date"].min(), sub["Date"].max())
            ax.set_title(f"{pt}  (n={n_total})", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=25)   # ← moved here, applies to ALL panels

            if ncols == 1 or (i % ncols == 0):
                ax.set_ylabel("Percent", fontsize=10)

        # Style unused panels
        for j in range(n, len(axes_list)):
            axes_list[j].set_facecolor("#f5f5f5")
            axes_list[j].set_axis_off()

        for ax in axes_list[:n]:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        legend_handles = [
            Line2D([0], [0], color="#DDB945", linewidth=2.5, marker="o", markersize=7, label="Strike %"),
            Line2D([0], [0], color="#AAAAAA", linewidth=2.5, linestyle="--", marker="s", markersize=6, label="Whiff %"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2,
           frameon=True, fancybox=True, edgecolor="#cccccc", fontsize=11)

        return fig

    @output
    @render.table
    def usage_table():
        data = pitcher_data()
        pid = input.player() if input.player_type() == "pitcher" else None
        session_type = input.session_type()

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

        if data is None or data.empty or not pid:
            return pd.DataFrame(
                columns=bullpen_cols if session_type == "bullpen" else live_scrimmage_cols
            )

        summary = compute_pitch_metrics(data, pid)
        if summary is None or summary.empty:
            return pd.DataFrame(
                columns=bullpen_cols if session_type == "bullpen" else live_scrimmage_cols
            )

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

        if session_type in {"live", "scrimmage"}:
            return out[live_scrimmage_cols]

        return out[live_scrimmage_cols]


app = App(app_ui, server, static_assets=STATIC_DIR)
