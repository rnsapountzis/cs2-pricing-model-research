# V8-Offset Candidate Re-Adjudication (latest)

**Setup.** Each candidate = **V8 raw logits + a no-intercept residual** from the added features
(keeps V8's discrimination, adds only the feature lift). All candidates calibrated with a
**common out-of-fold beta** recalibration for diagnostic re-adjudication only, not for production promotion, so the comparison
isolates the feature effect rather than a calibration-policy difference. n = 3,216 test rows.

Reference (raw V8 champion): **AUC 0.73065 · log loss 0.59471 · ECE10 0.01620.**

Δ = candidate better than the V8 champion under the common calibrator; `P(...)` are paired-bootstrap
probabilities (2,000 resamples for the deltas below use the same aligned rows). Full machine-readable
table: [`v8_offset_candidate_results.csv`](v8_offset_candidate_results.csv).

| candidate | Δ AUC | P(AUC↑) | Δ log loss | P(ll↑) | P(ECE worsens) | test coverage | note |
|---|---:|---:|---:|---:|---:|---:|---|
| **flash_csstats_pair** | **+0.00139** | 0.97 | **+0.00104** | 0.98 | 0.40 | 0.07 | strongest RC candidate |
| **csstats_pair** | +0.00075 | **1.00** | +0.00057 | **1.00** | 0.52 | 0.07 | most reliable (every bootstrap agrees) |
| csstats_rating30d_only | +0.00063 | **1.00** | +0.00046 | **1.00** | 0.69 | 0.07 | reliable, ECE borderline |
| flash_assists | +0.00076 | 0.86 | +0.00060 | 0.89 | 0.42 | 0.07 | borderline |
| flash_roi | +0.00072 | 0.85 | +0.00050 | 0.85 | 0.41 | 0.61 | borderline, high coverage |
| flash_all_research_selected | +0.00072 | 0.74 | +0.00080 | 0.84 | 0.40 | 0.06 | borderline |
| roster_history_counts | −0.00043 | 0.28 | −0.00005 | 0.46 | 0.35 | 0.66 | no gain on V8 base |
| roster_calibration_safe | −0.00012 | 0.39 | −0.00011 | 0.38 | 0.34 | 0.56 | no gain on V8 base |

## Interpretation

- **Grafted onto the true V8 base, the feature signal is real** — several candidates beat the champion
  on AUC *and* log loss with high bootstrap probability, without reliably worsening ECE. (An earlier
  V12 raw-retrain diagnostic did *not* beat V8, because it was built on a weaker retrained baseline;
  the correct experiment is the V8-offset residual above.)
- **`flash_csstats_pair`** is the strongest research candidate; **`csstats_pair`** is the most reliable
  (P = 1.00 on both AUC and log loss, though the absolute gain is smaller).
- **The synergy lives in CSStats player-form pairs × flash, not roster-overlap counts** — pure roster
  features are neutral-to-negative on the V8 base.
- **Gains are small** (log loss ≈ +0.0005–0.0010, AUC ≈ +0.0007–0.0014) — a model near its ceiling —
  but reliable and calibration-preserving.

## Status

`RC_candidate_for_forward_holdout` — **not** production-promoted. Every promotion is gated behind a
pre-registered forward holdout. See the coverage limitation below.

## Coverage limitation (the main blocker)

The CSStats residual features fire on only a minority of rows, and coverage **declines over time**
(machine-readable: [`csstats_coverage_by_split.csv`](csstats_coverage_by_split.csv)):

| split | rows | covered | coverage | covered date range |
|---|---:|---:|---:|---|
| train | 3,621 | 636 | 17.6% | 2023-09-29 → 2024-12-14 |
| valid | 4,522 | 518 | 11.5% | 2025-01-14 → 2025-12-21 |
| test | 3,216 | 225 | 7.0% | 2026-01-11 → 2026-06-20 |

A signal that fires on ~7% of recent rows cannot move the whole book, and its reliability on the
covered slice is limited by sample size. **Raising CSStats join coverage is the single biggest lever**
before any of these candidates could be considered for production promotion.
