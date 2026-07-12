# Model Results

The table below summarizes selected locked-test diagnostics from the research workspace.

Important: these are research diagnostics, not production promotion claims. The historical test set was used to diagnose multiple candidates, so any production-style promotion would require a fresh forward holdout.

| Model / policy | Log loss | AUC | Brier | ECE10 | Delta log loss | Delta AUC | Delta ECE10 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| V8 Champion, Strategy-D 45-55 | 0.594210 | 0.731161 | 0.205779 | 0.008098 | 0.000000 | 0.000000 | 0.000000 | Champion reference |
| V12 + flash ROI | 0.593572 | 0.731271 | 0.205547 | 0.009837 | -0.000638 | +0.000109 | +0.001739 | Research candidate |
| V12 + flash ROI + CSStats assists | 0.593345 | 0.731710 | 0.205451 | 0.011058 | -0.000865 | +0.000549 | +0.002961 | `diagnostic_only_not_promoted` |
| V12 + flash ROI + CSStats assists/rating30 | 0.592811 | 0.732768 | 0.205236 | 0.011931 | -0.001399 | +0.001606 | +0.003834 | `diagnostic_only_ece_reliability_risk` |
| Wide isotonic 30-70, Champion-only baseline | 0.614824 | 0.724022 | 0.211189 | 0.019681 | +0.020614 | -0.007139 | +0.011583 | Calibration overfit failure |

## Provenance note

The V12 combined CSStats rows are real raw-retrain diagnostics from the private workspace. They came from a separate V12 experiment that retrained the raw Champion feature list with selected new features and then re-ran fresh Champion-style calibration.

They are not promoted model claims. The separate CSStats recent-form bootstrap audit failed the project's reliability threshold, so CSStats features were classified as `parked_for_review_not_promoted` pending more coverage and forward validation.

## Interpretation

- Flash ROI produced the cleanest incremental feature signal.
- CSStats player-form combinations may contain signal, but the evidence was not strong enough for promotion.
- Directly optimizing validation ECE with a wide isotonic gate overfit badly.
- The original narrow Strategy-D 45-55 calibration remained the safer benchmark.

## Decision

No automatic promotion. Keep V8 as the champion reference. Treat flash ROI as a research candidate, and keep CSStats features `parked_for_review_not_promoted` until stronger bootstrap/forward evidence exists.
