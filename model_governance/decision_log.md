# Model Governance Decision Log

## V8 Champion

Status: champion reference / production-style baseline.

Configuration:

- Logistic regression, `C=0.05`.
- 71 raw features.
- Strategy-D gated calibration.
- Gate: `0.45-0.55`.
- Strategy-D fallback threshold: `0.035`.

Locked-test metrics:

- Log loss: `0.594210`
- AUC: `0.731161`
- Brier: `0.205779`
- ECE10: `0.008098`

Decision: keep unchanged as the reference model.

## V9/V12 flash ROI

Status: research candidate.

- `v9_flash_execution_roi_delta` showed incremental signal.
- V12 raw-retrain/fresh-calibration diagnostic:
  - Log loss: `0.593572`
  - AUC: `0.731271`
  - Brier: `0.205547`
  - ECE10: `0.009837`
  - Delta ECE10: `+0.001739`

Decision: useful research candidate; no production-style promotion without fresh forward holdout.

## CSStats recent-form reversal

Status: `parked_for_review_not_promoted`.

Initial observation:

- Recent-form features such as assists and rating deltas had attractive point estimates.
- Some V12 raw-retrain combinations improved log loss and AUC.

Reversal evidence:

- The separate CSStats residual/bootstrap audit did not clear the project's Step-8-style reliability bar.
- `rating30d_only` had weak ECE reliability despite an attractive point estimate.
- `assists_only` was more consistent but still did not meet the high reliability threshold required for promotion.
- Therefore no CSStats feature was frozen as an RC0/challenger.

Decision: keep CSStats features `parked_for_review_not_promoted` until coverage increases and the same bootstrap/forward criteria can be re-run.

## CSStats re-adjudication on the true V8 base (update — supersedes the reversal above)

Status: `RC_candidate_for_forward_holdout` (not production-promoted).

Context: the earlier CSStats parking was based on a standalone residual/bootstrap audit and on a V12
raw-retrain that sat on a **weaker retrained baseline**. The corrected experiment grafts the features
onto the **true V8 champion** as a no-intercept residual on the raw V8 logits, then re-adjudicates all
candidates under a **common out-of-fold beta calibrator** with paired-bootstrap CIs.

Result (locked test, vs raw V8 champion — see `reports/v8_offset_candidate_results.md`):

- `flash_csstats_pair`: ΔAUC `+0.00139` (P 0.97), Δlog loss `+0.00104` (P 0.98), ECE-neutral
  (P(ECE worsens) `0.40`). Upgraded to `RC_candidate_for_forward_holdout`.
- `csstats_pair`: smaller gains but P(improves) `= 1.00` on AUC and log loss. Reliable backup RC.
- Pure roster-overlap-count features remain neutral-to-negative on the V8 base; dropped from the
  promotion track.

Limitation (verified from data, `reports/csstats_coverage_by_split.csv`): CSStats features fire on only
~7% of test rows and coverage **declines over time** (train 17.6% → valid 11.5% → test 7.0%, spanning
2023-2026). This is a reliable but **low-coverage research candidate**, not a full historical Champion
upgrade.

Decision: upgrade `flash_csstats_pair` (primary) and `csstats_pair` (backup) to
`RC_candidate_for_forward_holdout`; **no production promotion** until the pre-registered forward holdout
passes and CSStats coverage is raised. V8 remains the Champion reference.

## V12 raw-retrain combined diagnostics

### `Champion71 + flash ROI + CSStats assists`

Status: `diagnostic_only_not_promoted`.

- Log loss: `0.593345`
- AUC: `0.731710`
- Brier: `0.205451`
- ECE10: `0.011058`
- Delta ECE10: `+0.002961`

Decision: evidence of possible signal, but not sufficient to override the CSStats bootstrap reversal.

### `Champion71 + flash ROI + CSStats assists/rating30`

Status: `diagnostic_only_ece_reliability_risk`.

- Log loss: `0.592811`
- AUC: `0.732768`
- Brier: `0.205236`
- ECE10: `0.011931`
- Delta ECE10: `+0.003834`

Decision: park for review; do not promote.

## Wide isotonic gate `0.30-0.70`

Status: rejected calibration policy.

Reason:

- Looked attractive on validation.
- Failed locked-test diagnostics.
- Treated as calibration overfit.

Decision: retain the original narrow Strategy-D 45-55 policy as the safer benchmark.

## Forward-validation requirement

Before any model is promoted:

- freeze feature list and calibration policy;
- pre-register sample-size gates;
- score genuinely future matches only;
- do not inspect performance metrics before sample gates are met;
- lock the first verdict to avoid repeated peeking.
