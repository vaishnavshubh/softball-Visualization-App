# Trackman Softball CSV — Schema & Context for Cursor

Use this document to give Cursor full context about the dataset so it can work on the Softball Pitch Dashboard without re-reading all CSV files.

---

## 1. Where the data lives

- **Path**: CSVs are under `app/data/v3/` (recursive). The app resolves this via `APP_DIR` so it works locally and on shinyapps.io.
- **Discovery**: `get_csv_paths()` globs `V3_PATH/**/*.csv` and returns sorted `(relative_path, full_path)`.
- **Date-range metadata**: `get_csv_paths_with_dates()` (or cached `.csv_metadata_cache.json`) gives per-file `(rel, full_path, dmin, dmax)` and global min/max dates. Each CSV is read once for the `Date` column to compute dmin/dmax; the app then loads only files whose date range overlaps the user-selected [start, end].

---

## 2. Columns the app uses (schema)

The app keeps only a subset of columns (`COLUMNS_TO_KEEP` in `app/app.py`). Below is the full list and how they’re used.

### Identifiers & metadata

| Column           | Type / notes        | Usage |
|-----------------|---------------------|--------|
| `PitchNo`       | int                 | Ordering / reference |
| `Date`          | date (parsed)       | Date-range filter; must exist for file to be included in metadata |
| `Time`          | time                | Kept only; not used in logic |
| `PitchofPA`     | int                 | Kept only |
| `Pitcher`       | string              | Display name; cleaned with strip(); "Last, First" → "First Last" via `format_pitcher_display_name()` |
| `PitcherId`     | int/str             | Primary key for pitcher; used in filters and dropdowns |
| `PitcherThrows` | string              | "Left" / "Right"; shown in header and in matchup splits |
| `PitcherTeam`   | string              | Team filter; populates Team dropdown (Pitcher view) |
| `Batter`        | string              | Display name; same cleaning as Pitcher |
| `BatterId`       | int/str             | Primary key for batter; used in Hitter view and matchup splits |
| `BatterSide`     | string              | "Left" / "Right"; used in pitcher vs L/R and batter vs L/R splits |
| `BatterTeam`     | string              | Team filter in Hitter view |
| `Balls`         | int                 | Count; used with `Strikes` to build "Balls-Strikes" (e.g. "0-0") for pitch mix by count |
| `Strikes`       | int                 | Count; see `Balls` |

### Pitch type & outcome

| Column             | Type / notes   | Usage |
|--------------------|----------------|--------|
| `TaggedPitchType`  | string         | Primary pitch-type label; cleaned with strip(). Used for usage %, colors, location/movement by type, and all pitch-type tables. Fixed color map in `PITCH_TYPE_FIXED_COLORS` (e.g. Fastball, Changeup, Curveball, Riseball, Dropball, Screwball, Offspeed, Riser, Drop, Rise). |
| `PitchCall`        | string         | Outcome: e.g. "StrikeCalled", "InPlay", "Foul", "SwingingStrike", "FoulTip". Used to derive swing (`is_swing`) and whiff (`is_whiff`) for zone/swing/whiff stats and splits. |
| `TaggedHitType`    | string         | e.g. "FlyBall"; kept when present |
| `PlayResult`       | string         | e.g. "Out"; kept when present |

### Velocity, spin, movement

| Column             | Type / notes   | Usage |
|--------------------|----------------|--------|
| `RelSpeed`         | float          | mph; used for avg velocity in header and in pitcher stats-by-type table |
| `SpinRate`         | float          | rpm; pitcher stats-by-type |
| `SpinAxis`         | float          | kept |
| `Tilt`             | string/float   | kept |
| `InducedVertBreak` | float          | inches; Y axis on movement plot; pitcher stats-by-type as "VB" |
| `HorzBreak`        | float          | inches; X axis on movement plot; pitcher stats-by-type as "HB" |

### Location (plate)

| Column           | Type / notes  | Usage |
|------------------|---------------|--------|
| `PlateLocHeight` | float         | feet; vertical location at plate; used with `PlateLocSide` for strike-zone and in-zone logic |
| `PlateLocSide`   | float         | feet; horizontal location at plate |

Strike zone (feet): `ZONE_LEFT = -0.83`, `ZONE_RIGHT = 0.83`, `ZONE_BOTTOM = 1.5`, `ZONE_TOP = 3.5`. In-zone = both coordinates within these bounds; used for Zone% and chase logic.

### Batted ball (when available)

| Column    | Type / notes | Usage |
|-----------|--------------|--------|
| `ExitSpeed` | float      | mph; batter exit velocity; batter stats and EV histogram |
| `Angle`     | float      | launch angle; kept |
| `Direction`  | float     | kept |

### Release (when available)

| Column     | Type / notes | Usage |
|------------|--------------|--------|
| `RelHeight` | float       | kept |
| `RelSide`   | float       | kept |

### Trajectory coefficients

Pitch trajectory: `PitchTrajectoryXc0/1/2`, `PitchTrajectoryYc0/1/2`, `PitchTrajectoryZc0/1/2` — kept for compatibility; not used in current analytics.

---

## 3. Derived concepts used in the app

- **Swing**: `PitchCall` in `("inplay", "foul", "swingingstrike", "foultip", "foulball")` (case-insensitive).
- **Whiff**: `PitchCall == "swingingstrike"`.
- **In-zone**: `PlateLocSide` and `PlateLocHeight` both within strike-zone bounds above; NaN → not in zone.
- **Count string**: `"{Balls}-{Strikes}"` (e.g. "0-0", "3-2") for pitch mix by count.
- **Pitcher display name**: "Last, First" → "First Last".
- **Date range**: User picks start/end date; app loads every CSV whose `[dmin, dmax]` overlaps `[start, end]`, then filters rows to `Date in [start, end]`.

---

## 4. File layout and loading

- One CSV per file; multiple CSVs can be combined when the selected date range spans files.
- `load_and_clean_csv(full_path)` keeps only `COLUMNS_TO_KEEP` and strips `Pitcher`, `Batter`, `TaggedPitchType`, `PitchCall`.
- For date-range selection, only the `Date` column is read to build metadata (or cache); full CSVs are loaded only when needed for the selected range.

---

## 5. Summary for Cursor

- **Primary keys**: `PitcherId` (pitcher), `BatterId` (batter). Filter by `PitcherTeam` or `BatterTeam` for dropdowns.
- **Pitch type**: `TaggedPitchType`; fixed colors; usage %, location, and movement are all by this column.
- **Outcomes**: `PitchCall` drives swing/whiff; optional `ExitSpeed`/`Angle`/`Direction` for batted-ball views.
- **Location**: `PlateLocSide`, `PlateLocHeight` in feet; zone constants in `app.py`.
- **Movement**: `HorzBreak`, `InducedVertBreak` in inches.
- **Dates**: Parsed from `Date`; filtering is inclusive [start, end] on calendar date.

Share this document with your friend so Cursor on her machine has full dataset context without re-reading the CSVs.
