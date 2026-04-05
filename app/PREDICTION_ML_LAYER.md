# Prediction Tab Logic and ML Layer

This document explains how the Prediction tab works in the softball dashboard, including:

- the descriptive (non-ML) logic,
- the ML training/scoring pipeline,
- fallback behavior when data is limited,
- and how recommendations are generated.

## 1) Scope and Entry Points

Prediction is currently **pitcher-only** in the UI. The tab is built from these core server reactives and functions:

- `prediction_df()` -> `build_prediction_by_pitch_type(data)`
- `prediction_summary()` -> `select_prediction_summary(prediction_df())`
- `ml_prediction_state()` -> `compute_ml_prediction_bundle(data, train_pool=all_data())`
- `prediction_content()` and `prediction_table()` render either ML or descriptive outputs

High-level flow:

1. Build per-pitch descriptive metrics for the selected pitcher.
2. Try to train and score ML models.
3. If ML succeeds, render ML recommendation cards and ML table.
4. If ML fails or sample is too small, fall back to descriptive recommendation cards and table.

---

## 2) Descriptive Layer (Always Available if Data Exists)

Primary function: `build_prediction_by_pitch_type(df)`.

For each pitch type, it computes:

- `pitch_count`
- `strike_pct`
- `swing_count`
- `whiff_count`
- `contact_count`
- `contact_pct`
- `whiff_pct` (`whiff_count / swing_count`)
- `hard_contact_count`
- `hard_contact_risk` (`hard_contact_count / contact_count`)
- `sample_warning` (small-sample warning text)

### Event definitions

- **Swing events**: `StrikeSwinging`, `FoulBallFieldable`, `FoulBallNotFieldable`, `InPlay`
- **Strike events**: `StrikeCalled`, `StrikeSwinging`, `FoulBallFieldable`, `FoulBallNotFieldable`, `InPlay`
- **Contact events**: `FoulBallFieldable`, `FoulBallNotFieldable`, `InPlay`
- **Descriptive hard contact threshold**: exit speed >= `85 mph` (`PREDICTION_HARD_CONTACT_EV`)

### Descriptive recommendation selection

Function: `select_prediction_summary(pred)`.

It chooses:

- **Best strike pitch**: highest `strike_pct` with tiered minimum volume checks
- **Best put-away pitch**: highest `whiff_pct` with tiered swing-count minimums
- **Caution pitch**: highest `hard_contact_risk` with tiered contact minimums

Tiered thresholds:

- strike candidate preferred minimum: `pitch_count >= 12`, fallback `>= 6`, then any
- put-away candidate preferred minimum: `swing_count >= 8`, fallback `>= 4`, then any with swings
- caution candidate preferred minimum: `contact_count >= 5`, fallback `>= 2`

---

## 3) ML Layer Overview

Primary function: `compute_ml_prediction_bundle(df, train_pool=None)`.

The ML module fits **three separate binary logistic regression models**:

1. Strike probability model
2. Whiff probability model
3. Hard-contact probability model

The output is a bundle:

- `use_ml` (bool)
- `message` (fallback reason when ML is not used)
- `df` (per-pitch-type predicted probabilities)
- `summary` (best strike, best put-away, caution selections from ML outputs)
- `training_note` (notes when training was widened beyond pitcher-only data)

If `scikit-learn` is unavailable or training conditions are not met, `use_ml=False` and the tab falls back to descriptive logic.

---

## 4) ML Features and Targets

### Features

Configured feature sets:

- Numeric (`ML_NUMERIC_FEATURES`):
  - `PlateLocSide`
  - `PlateLocHeight`
  - `Balls`
  - `Strikes`
  - `RelSpeed`
  - `SpinRate`
  - `InducedVertBreak`
  - `HorzBreak`
- Categorical (`ML_CATEGORICAL_FEATURES`):
  - pitch type column (`PITCH_TYPE_COL`)
  - `BatterSide`
  - `PitcherThrows`

Only columns that actually exist in the data are used.

### Targets

Built in `prepare_pitcher_ml_training_frame(df)`:

- `y_strike`: pitch event is in strike events set
- `y_whiff`: pitch event is exactly `StrikeSwinging`
- `y_hard`: `InPlay` and exit speed >= `80 mph` (`ML_HARD_CONTACT_EV`)

Note: ML hard-contact threshold (`80 mph`) is intentionally different from descriptive threshold (`85 mph`).

---

## 5) Training Gates and Small-Sample Behavior

Key constants:

- `ML_MIN_TRAIN_ROWS = 30`
- `ML_MIN_PER_CLASS = 1`

Checks:

1. At least 30 usable rows after pitch-type filtering.
2. Each target has at least two classes and class counts meeting minimum constraints.

To avoid single-class training failures, `ensure_binary(y)` flips one row if needed so logistic regression can fit. This is a robustness guardrail for sparse edge cases.

### Widened training pool

If pitcher-level usable rows are below 30 and a broader filtered dataset is available:

- model training uses `train_pool` (all pitches under current filters, across pitchers),
- scoring still uses pitcher-specific representative rows,
- `training_note` is shown in UI to disclose this behavior.

---

## 6) Preprocessing and Model Configuration

Function: `_fit_logistic_pipeline(X, y)`.

Pipeline architecture:

1. `ColumnTransformer`
2. Numeric branch:
   - `SimpleImputer(strategy="median")`
   - `StandardScaler()`
3. Categorical branch:
   - `SimpleImputer(strategy="most_frequent")`
   - `OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=25)`
4. Classifier:
   - `LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42, solver="lbfgs")`

This setup is designed for:

- stable training on mixed feature types,
- resilience to missing values,
- and better minority-class handling (`class_weight="balanced"`).

---

## 7) How Per-Pitch Predictions Are Scored

Function: `build_typical_pitch_feature_rows(df, num_cols, cat_cols)`.

For each pitch type, one representative ("typical") row is created:

- numeric features = mean within that pitch type
- `BatterSide` = mode of the pitcher sample
- `PitcherThrows` = mode of the pitcher sample
- pitch type stays fixed to that pitch

Then each of the 3 trained models produces `predict_proba(...)[,1]` for each typical row:

- `predicted_strike_prob`
- `predicted_whiff_prob`
- `predicted_hard_contact_prob`

Sample warnings are attached, and an extra warning is added for small pitch-type sample (`pitch_count < 20`):

- `"Low sample — ML estimate may be unstable"`

---

## 8) ML Recommendation Logic

Function: `select_ml_prediction_summary(pred)`.

It chooses one pitch for each recommendation card:

- **Best strike**: max `predicted_strike_prob`
- **Best put-away**: max `predicted_whiff_prob`
- **Caution**: max `predicted_hard_contact_prob`

With tiered sample filters:

- strike tier: `pitch_count >= 12`, fallback `>= 6`, then any
- put-away tier: `pitch_count >= 8`, fallback `>= 6`, then any
- caution tier: `pitch_count >= 5`, fallback `>= 3`, then any

The table tags each pitch with recommendation labels:

- `Best strike`
- `Best put-away`
- `Caution`

---

## 9) Fallback and User Messaging

If ML cannot be used, UI shows descriptive cards/table and a message such as:

- sklearn not installed
- insufficient training rows
- model fitting failure
- prediction/scoring failure

If ML is used with broadened training data, UI shows `training_note` to explain that models were fit on a wider sample.

---

## 10) Interpretation Guidance

- ML outputs are **estimated probabilities** under a typical feature profile, not guarantees.
- Descriptive outputs are **historical rates** under current filters.
- Small-sample warnings should be treated as reliability flags.
- Caution pitch highlights hard-contact risk, not necessarily "never throw this pitch."

---

## 11) Key Implementation Functions (Quick Reference)

- `build_prediction_by_pitch_type`
- `select_prediction_summary`
- `prepare_pitcher_ml_training_frame`
- `_fit_logistic_pipeline`
- `build_typical_pitch_feature_rows`
- `compute_ml_prediction_bundle`
- `select_ml_prediction_summary`
- `format_prediction_table_display`
- `format_ml_prediction_table_display`

