# Prediction Tab Guide

This guide explains how the **Prediction** tab works, what the model outputs mean, and how to interpret recommendations in practice.

## What The Prediction Tab Does

The Prediction tab ranks a pitcher's pitch types for a selected filter context (date range, team, pitcher, batter-side filter) and shows:

- A **Pitching Strategy Summary** sentence at the top
- **Best strike / command pitch**
- **Best put-away option**
- **Highest damage-risk pitch**
- A simplified **summary table** by pitch type for quick comparison
- An optional **Advanced Metrics** section (collapsed by default)

The goal is decision support, not certainty. Outputs are **model estimates** and should be combined with game plan, scouting, and catcher/pitcher feedback.

## Data Scope

- Training data is built from the **full Trackman pool** available to the app (league-wide rows in current data load), optionally narrowed by batter-side filter (`Combined`, `vs Right`, `vs Left`) for matchup context.
- Inference is done for the **currently selected pitcher** using that pitcher's own shape/location profile per pitch type.

This design gives stronger training signal than player-only training while still producing pitcher-specific recommendations.

## Model Pipeline (High-Level)

Implemented in `app/prediction_pipeline.py`.

1. Build pitch-level labels:
   - **Strike**: strike-like outcomes
   - **Whiff**: swinging strike
   - **Hard contact risk**: in-play EV >= configured threshold
2. Engineer features (when available):
   - pitch type, velocity, spin, movement, location, count
   - previous pitch type / previous velocity / velocity delta
   - zone flags (`in_zone`, upper/lower, inner/outer)
   - count-state flags (`is_two_strike`, hitter-count, pitcher-count)
   - platoon context
3. Add smoothed batter/pitcher priors:
   - batter vs pitch-type tendencies
   - pitcher pitch-type tendencies + usage
   - empirical-Bayes style smoothing toward global pitch-type priors
4. Train one binary model per target (`strike`, `whiff`, `hard`):
   - preference order: **XGBoost -> LightGBM -> sklearn HistGradientBoosting**
   - probability calibration (isotonic) where trainable
5. Predict probabilities for one synthetic profile row per pitch type for the selected pitcher:
   - neutral count (for command/damage)
   - two-strike simulation (for put-away context)
6. Convert probabilities into decision scores and rank recommendations.

## Recommendation Logic

Recommendations are **not** simple max-probability picks. They use weighted composite scores:

- **Command score**: favors high strike probability and penalizes hard-contact risk
- **Put-away score**: favors high whiff in two-strike context and penalizes hard-contact risk
- **Damage score**: highlights pitch types with elevated hard-contact risk (and weaker strike reliability)

Weights are configurable at the top of `app/prediction_pipeline.py`.

In the UI, these scores are intentionally de-emphasized for coaches and moved to **Advanced Metrics**.

## Coach-First Language

The Prediction tab now uses short coaching language:

- brief one-line takeaways
- short plain-language explanation
- no heavy analytics wording in the main cards

Examples:

- "Most reliable pitch to get ahead in the count"
- "Best swing-and-miss pitch in two-strike situations"
- "Most likely to get hit hard if mislocated"

## Fallback And Stability Guardrails

To avoid brittle behavior on sparse outcomes:

- If class balance is too weak for a target (e.g., very few hard-contact positives), that target can fall back to a **prior-probability model**.
- The UI shows a **Model fallback note** when this happens.
- Each pitch row and card includes sample/stability context.

Interpretation rule:

- **Stable/Moderate** sample labels -> more trust
- **Low/Very low** sample labels -> treat as directional only

## How To Read The Cards

For each card, read top-to-bottom:

1. Pitch type recommendation
2. Primary takeaway (coach-facing)
3. Predicted percentages:
   - strike %
   - whiff %
   - hard-contact risk %
4. Short plain-language explanation
5. "When to use" bullet points
6. Sample note (if relevant)

Color cues:

- **Green**: stronger / safer option
- **Yellow**: situational option
- **Red**: higher risk option

Suggested usage:

- Use **Best strike / command** in neutral or must-strike situations.
- Use **Best put-away** in finish counts (two-strike lean).
- Treat **Highest damage-risk** as a pitch/location combo to manage more carefully, not to ban entirely.

## How To Read The Summary Table

Columns include:

- Pitch
- Strike %
- Whiff %
- Hard Contact Risk
- Role (`Command`, `Put-away`, `Risk`, or `Situational`)
- Recommendation

Use it to compare second-best options, not just the top card picks.

If you need deeper score details, open **Advanced Metrics**.

## Advanced Metrics UI Spec (SHAP-Friendly)

Goal: keep coach workflow simple while preserving analyst depth.

### 1) Section framing

- Keep **Advanced Metrics** collapsed by default.
- Add a helper line under the section title:
  - "Model detail view: shows how each pitch profile shifts expected outcomes versus league-average baseline."
- Add an inline legend:
  - **Green/right** = increases the outcome
  - **Red/left** = decreases the outcome
  - "Values are directional effects, not raw probabilities."

### 2) Plain-language labels (replace score jargon)

- `Command score` -> **Control impact**
- `Put-away score` -> **Two-strike put-away impact**
- `Damage score` -> **Hard-contact risk impact**
- `Composite` -> **Overall profile impact**

If technical names are needed for analysts, show them in muted parentheses:
- "Control impact (command score)"

### 3) Information hierarchy

Within Advanced Metrics, order content as:

1. **Top 3 impact chips** (single line, quick read)
   - Example: "Dropball: strongest control impact (+1.07)"
   - Example: "Changeup: strongest put-away impact (+1.20)"
   - Example: "Fastball: highest hard-contact risk impact (+0.51)"
2. **Per-pitch impact bars** (primary visualization)
3. **Detailed numeric table** (secondary, analyst use)

### 4) Per-pitch impact bars (primary visual)

For each pitch type, show three horizontal bars:

- Control impact
- Two-strike put-away impact
- Hard-contact risk impact

Behavior:

- Centered zero-line with left (negative) and right (positive) fill.
- Signed values visible at right edge of each bar (`+1.07`, `-0.59`).
- Consistent axis scale within the panel so users can compare pitches.
- Sort pitches by `Overall profile impact` descending.

Color guidance:

- Control / put-away bars: positive = green, negative = red.
- Risk bars: positive (more risk) = red, negative (less risk) = green.

### 5) Numeric table (secondary detail)

Keep the existing table, but:

- Use the plain-language column names above.
- Right-align numeric columns and cap precision at 3 decimals.
- Default sort by `Overall profile impact` descending.
- Add a small note under table:
  - "Larger absolute values indicate stronger model influence in this filter context."

### 6) Coach-safe interpretation copy

Use this exact helper text at the bottom of Advanced Metrics:

"Use these values to compare relative strengths across this pitcher's options. Treat small gaps as equivalent tiers; prioritize cards and game context for final pitch calling."

### 7) Display rules for low-sample contexts

When sample stability is `Low` or `Very low`:

- Keep bars visible but reduce emphasis (muted alpha).
- Show warning badge: "Low sample: directional only."
- Suppress fine-grained ranking language ("best", "worst") in the chip row and use "leans toward" wording.

### 8) Accessibility and readability requirements

- Minimum 4.5:1 text contrast for labels and signed values.
- Do not rely on color alone: keep plus/minus signs and zero-line.
- Ensure section is readable at common laptop zoom (100%); avoid multi-line metric labels where possible.

## Visual Walkthrough (Coach View)

Use this as a quick tour when presenting the tab to staff:

1. **Top strategy box ("Pitching Strategy Summary")**
   - Read this first.
   - It gives the game-plan sentence in plain language:
     - control pitch
     - put-away pitch
     - risk pitch

2. **Three recommendation cards**
   - Start with the large pitch name.
   - Read the bold one-line takeaway.
   - Check only the 3 percentages (Strike, Whiff, Hard Contact Risk).
   - Use "When to use" bullets for in-game decisions.

3. **Color meaning on cards**
   - Green = safer/stronger option
   - Yellow = situational option
   - Red = higher risk if execution misses

4. **Summary by pitch type table**
   - Use for side-by-side comparison across all pitch types.
   - Focus on `Role` to quickly identify Command / Put-away / Risk.

5. **Advanced Metrics (optional)**
   - Open only for analyst review.
   - Keep closed during coach-first conversations.

### Suggested Screenshot Callouts

If you are building slides or internal docs, add labels directly on screenshots:

- **A. Strategy Summary** -> "Start here"
- **B. Command Card** -> "Go-to strike pitch"
- **C. Put-away Card** -> "Two-strike finisher"
- **D. Risk Card** -> "Damage-risk warning"
- **E. Summary Table** -> "Compare all pitch types quickly"
- **F. Advanced Metrics** -> "Analyst-only detail"

## Operational Notes

- Models are cached for responsiveness.
- Use the **Retrain Prediction Models** button to clear cache and force fresh training on next refresh (helpful after data updates).
- First run after retrain may take longer; subsequent interactions are faster.

## Interpretation Caveats

- Outputs are probability estimates, not deterministic outcomes.
- Recommendations depend on selected filters and available data quality.
- Hard-contact estimates can be noisy in small in-play samples.
- Always pair model output with context: pitcher feel, opponent approach, game state, and scouting.

## Quick Coach-Friendly Summary

- Think of Prediction as a **game-plan board**, not a math report.
- Use the top strategy sentence first, then the three cards.
- Use the table for quick side-by-side checks across pitch types.
- Use Advanced Metrics only when analysts need deeper scoring detail.
