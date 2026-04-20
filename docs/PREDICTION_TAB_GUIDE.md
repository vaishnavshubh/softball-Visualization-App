# Prediction Tab Guide

Last updated: 2026-04-20

This guide documents the current behavior of the **Prediction** tab in the app, including ML mode, fallback behavior, and how to interpret each section.

## Availability And Scope

Prediction is shown only when:

- **Player Type** is `Pitcher`
- **Data Source** is `Trackman`

If those conditions are not met, the tab shows a message instead of recommendations.

Model training uses league-wide Trackman data (optionally filtered by batter side), then generates pitcher-specific recommendations from that trained model.

## What The Tab Shows

When prediction data is available, the tab renders:

- A **Pitching Strategy Summary** sentence
- Three recommendation cards:
  - **Best strike / command pitch**
  - **Best put-away option**
  - **Highest damage-risk pitch**
- A **Summary by pitch type** table
- An **Advanced Metrics** section (collapsed by default, shown when ML mode is active)

## Recommendation Cards (Coach View)

Each card includes:

- Pitch name
- One-line coaching takeaway
- Predicted `Strike`, `Whiff`, and `Hard Contact Risk` percentages
- Short plain-language explainer
- Two "When to use" bullets
- Sample warning (when relevant)

Card colors are role-based:

- Green: command recommendation
- Gold/yellow: put-away recommendation
- Red: damage-risk caution recommendation

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

## Summary Table Behavior

### ML table columns

- `Pitch`
- `Strike %`
- `Whiff %`
- `Hard Contact Risk`
- `Role` (`Command`, `Put-away`, `Risk`, `Situational`)
- `Recommendation`

## EV Interpretation Bands (Coach Context)

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
2. A color legend:
   - Green/right = increases the metric
   - Red/left = decreases the metric
3. Top impact chips:
   - strongest control impact
   - strongest two-strike put-away impact
   - highest hard-contact risk impact
4. Per-pitch impact cards with bars for:
   - `Control impact`
   - `Two-strike put-away impact`
   - `Hard-contact risk impact`
5. **Model Drivers (SHAP)** cards for best command / best put-away / highest risk
6. Advanced numeric table with:
   - `Control impact`
   - `Two-strike put-away impact`
   - `Hard-contact risk impact`
   - `Overall profile impact`

Low-sample stability behavior in Advanced Metrics:

- Low/very-low rows are visually muted
- A badge appears: `Low sample: directional only.`
- Impact chip language softens to `leans toward...`

## Notes Shown Above Cards (ML Mode)

When available, ML mode also shows:

- **Training note** (what data/training scheme was used)
- **Validation note** (ROC-AUC summary by target)
- **Estimate note** (model warnings/fallback internals from training bundle)

## Retraining And Cache

- Models are cached (memory + disk) for faster repeated use.
- The **Retrain Prediction Models** button clears prediction cache.
- After retrain, the next refresh runs training again and may take longer.

## Interpretation Guidelines

- Use this as decision support, not a deterministic answer engine.
- Treat low-sample outputs as directional.
- Always combine tab recommendations with game context, sequencing plan, and pitcher feel.
