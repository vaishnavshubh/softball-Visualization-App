# Prediction Tab Guide

Last updated: 2026-04-23

This guide documents the current behavior of the **Prediction** tab in the app, including ML mode, fallback behavior, and how to interpret each section.

## Availability And Scope

Prediction is shown only when:

- **Player Type** is `Pitcher`
- **Data Source** is `Trackman`

If those conditions are not met, the tab shows a message instead of recommendations.

Model training uses league-wide Trackman data (optionally filtered by batter side), then generates pitcher-specific recommendations from that trained model.

## What The Tab Shows

When prediction data is available, the tab renders:

- A **Pitching Development Summary** sentence
- Three recommendation cards:
  - **Best strike / command development pitch**
  - **Best two-strike development option**
  - **Highest damage-risk pitch to address**
- A **Summary by pitch type** table
- An **Advanced Metrics** section (collapsed by default, shown when ML mode is active)

## Recommendation Cards (Player Development View)

Each card includes:

- Pitch name
- One-line player-development takeaway
- Predicted `Strike`, `Whiff`, and `Hard Contact Risk` percentages
- Short plain-language explainer
- Two "Player development focus" bullets
- Sample warning (when relevant)

Card colors are role-based:

- Green: command-development recommendation
- Gold/yellow: two-strike development recommendation
- Red: damage-risk area to address

## Prediction Modes

The tab supports two output modes:

1. **ML mode (preferred)**  
   Uses `prediction_pipeline.compute_ml_prediction_bundle(...)` with league-trained models and pitcher profile rows.

2. **Descriptive fallback mode**  
   Used when ML cannot run (missing dependencies, insufficient league rows, no trainable balance, etc.).  
   In this mode, cards and table are built from pitcher-level descriptive rates instead of ML probabilities.

When fallback is active, the UI shows a fallback note and does not render the ML-only advanced section.

## ML Pipeline (Current High-Level)

Implemented in `app/prediction_pipeline.py`.

1. Filter league pool by batter-side context (`Combined`, `vs Right`, `vs Left`)
2. Engineer pitch-level features (shape, location, count state, previous pitch context, platoon, zone flags)
3. Add smoothed batter/pitcher priors by pitch type
4. Train/load one model per target:
   - strike
   - whiff
   - hard-contact risk
5. Predict by pitch type using representative pitcher profiles for:
   - neutral count context (command + risk)
   - two-strike context (put-away)
6. Compute decision scores:
   - `attack_score`
   - `putaway_score`
   - `danger_score`
   - `composite_score`
7. Select the three recommendation cards using score + sample guardrails

## How These Values Are Calculated

This section explains where Advanced Metrics values come from:
`Strike reliability`, `Put-away value`, and `Hard-contact risk`.

### Step-by-step pipeline

1. **Raw data**
   - League-wide Trackman pitch-by-pitch data is the training pool.
   - The current pitcher selection supplies the profile rows that get scored.
2. **Feature engineering**
   - The model uses pitch shape, velocity, location, count, sequence context, handedness, and smoothed tendency features.
3. **Model predictions**
   - Three gradient-boosted models estimate:
     - strike probability
     - whiff (swing-and-miss) probability
     - hard-contact probability
4. **Contextual profiles**
   - Predictions are generated from representative pitch profiles (not a single raw row):
     - neutral-count context (used for strike reliability and risk context)
     - two-strike context (used for put-away context)
5. **Decision scores**
   - Model probabilities are transformed into score space to support side-by-side decision use:
     - `Strike reliability` is derived from predicted strike probability (with risk-aware adjustments)
     - `Put-away value` is derived from predicted whiff probability, especially in two-strike context
     - `Hard-contact risk` is derived from predicted hard-contact probability
6. **Displayed impact values**
   - Values shown in Advanced Metrics are **relative impact scores** versus a league-average baseline for the same modeling context.
   - These scores drive chips, bars, and ranking table values.

### Why these are relative impact values

- They are designed for **comparison across pitch types**, not for standalone prediction reporting.
- They preserve direction and strength so coaches can prioritize pitch usage and development focus quickly.

## What The Numbers Mean

Advanced Metrics values are:

- **Not raw probabilities**
- **Not percentages**
- **Relative model-derived effect sizes**

Quick interpretation:

- `+0.8` -> strong positive impact versus baseline
- `0.0` -> near baseline
- negative -> below baseline for that outcome

Important nuance for `Hard-contact risk`:

- lower / more negative = safer (less expected hard-contact damage)
- higher / closer to zero or positive = more damage risk

## How To Read The Bars

- Bars are a visual translation of the same impact values shown numerically.
- Right side = stronger positive effect for that metric.
- Left side = weaker/negative effect for that metric.
- For `Hard-contact risk`, left is safer and right is more dangerous.
- Inline cue in the UI (`Safer <-> More damage risk`) is there to make risk direction obvious at a glance.

## Why Not Just Show Probabilities?

Raw model probabilities are calculated internally, but Advanced Metrics is designed for decision support across multiple goals at once:

- control (strike execution)
- put-away ability
- damage prevention risk

To compare those outcomes consistently, the app converts model outputs into standardized relative impact scores.  
This makes pitch-to-pitch comparison, ranking, and planning more actionable than reading three separate probability columns alone.

## Limitations And Proper Use

- Small samples increase uncertainty; estimates rely more on league priors in those cases.
- Scores are relative to model baseline context, not absolute guarantees.
- Context still matters: count state, sequencing, location quality, and opponent tendencies.
- Use these values as guidance for planning and development, not as exact outcome predictions.

## Summary Table Behavior

### ML table columns

- `Pitch`
- `Strike %`
- `Whiff %`
- `Hard Contact Risk`
- `Role` (`Command`, `Put-away`, `Risk`, `Situational`)
- `Recommendation`

## EV Interpretation Bands (Player Development Context)

Hard-contact interpretation is calibrated to typical D1 softball ranges:

- `70-75 mph`: typical contact
- `75-80 mph`: firm contact
- `80+ mph`: high-damage contact

In descriptive fallback mode, `Hard Contact Risk` reflects a weighted damage profile:

- firm contact contributes directional risk
- `80+ mph` contributes primary damage risk

### Descriptive fallback table columns

- `Pitch`
- `Strike %`
- `Whiff %`
- `Contact %`
- `Hard Contact Risk`
- `Sample`
- `Sample Note`
- `Recommendation`

## Advanced Metrics (ML Mode)

The Advanced section includes:

1. Helper copy describing model-directional impacts vs baseline
2. An impact key:
   - Right (green) = helps that outcome
   - Left = hurts that outcome
   - For hard-contact risk: lower (left) = safer, higher (right) = more damage risk
3. Top impact chips:
   - strongest strike reliability
   - strongest put-away value
   - highest hard-contact risk
4. Per-pitch impact cards with bars for:
   - `Strike reliability`
   - `Put-away value`
   - `Hard-contact risk`
   - Risk row includes inline cue: `Safer <-> More damage risk`
   - Risk row includes visible helper text: `Lower is better (less hard contact allowed).`
5. **Why each pitch grades this way** cards with:
   - `What's helping this pitch`
   - `What could hurt this pitch`
6. Advanced numeric table with:
   - `Strike reliability`
   - `Put-away value`
   - `Hard-contact risk`
   - `Overall decision value` (combined ranking score, not an additional probability metric)

Low-sample stability behavior in Advanced Metrics:

- Low/very-low rows are visually muted
- A badge appears: `Low sample: trend only.`
- Impact chip language softens to `leans toward...`

## Notes Shown Above Cards (ML Mode)

When available, ML mode also shows:

- **Training note** (what data/training scheme was used)
- **Model validation note** (ROC-AUC summary by target)
- **Model estimate note** (model warnings/fallback internals from training bundle)

## Retraining And Cache

- Models are cached (memory + disk) for faster repeated use.
- The **Retrain Prediction Models** button clears prediction cache.
- After retrain, the next refresh runs training again and may take longer.

## Interpretation Guidelines

- Use this as player-development decision support, not a deterministic answer engine.
- Treat low-sample outputs as directional.
- Always combine tab recommendations with game context, sequencing plans, pitcher feel, and current development priorities.
