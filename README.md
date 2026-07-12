# CS2 Pricing Model Research

Calibration-aware CS2 map-winner modelling project focused on pricing accuracy, feature validation, and production-safe model governance.

This repository is a polished portfolio extract from a larger private research workspace. It intentionally includes methodology, reusable code patterns, result summaries, and governance notes -- not raw scraped datasets, HAR files, or private data dumps.

## Why this project exists

The goal was to simulate the kind of analytical work required in esports betting and trading:

- build and evaluate CS2 map-winner pricing signals;
- identify model inefficiencies without fooling ourselves;
- test new feature families under strict train/valid/test discipline;
- monitor log loss, AUC, Brier score, and ECE10 calibration;
- reject or park features that improve ranking but damage probability calibration;
- document champion/challenger decisions in a way that traders and technical reviewers can both understand.

The project is not framed as a "profitable betting bot". It is a research-grade model validation and pricing-analysis workflow.

## What this project found

The existing V8 Champion model was a calibrated logistic-regression map-winner model. The strongest defensible finding was not "deploy a bigger model"; it was:

1. Flash execution ROI produced real incremental signal when tested as a raw V12 retrain feature.
2. Some CSStats player-form combinations also improved log loss/AUC in a V12 raw-retrain diagnostic, but they were not promoted because separate bootstrap reliability checks did not clear the project's evidence bar.
3. Calibration risk, especially ECE10, was the binding constraint.

| Model / policy | Log loss | AUC | Brier | ECE10 | Delta log loss | Delta AUC | Delta ECE10 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| V8 Champion, Strategy-D 45-55 | 0.594210 | 0.731161 | 0.205779 | 0.008098 | 0.000000 | 0.000000 | 0.000000 | Champion reference |
| V12 + flash ROI, Strategy-D 45-55 | 0.593572 | 0.731271 | 0.205547 | 0.009837 | -0.000638 | +0.000109 | +0.001739 | Research candidate |
| V12 + flash ROI + CSStats assists, Strategy-D 45-55 | 0.593345 | 0.731710 | 0.205451 | 0.011058 | -0.000865 | +0.000549 | +0.002961 | `diagnostic_only_not_promoted` |
| V12 + flash ROI + CSStats assists/rating30, Strategy-D 45-55 | 0.592811 | 0.732768 | 0.205236 | 0.011931 | -0.001399 | +0.001606 | +0.003834 | `diagnostic_only_ece_reliability_risk` |

Important interpretation:

- The V12 combined CSStats rows are real raw-retrain diagnostics from the private workspace, but they are not promotion claims.
- Earlier CSStats residual/bootstrapped tests failed the project's reliability bar, so CSStats features remained `parked_for_review_not_promoted` pending more coverage and forward evidence.
- The most portfolio-relevant lesson is the governance discipline: an attractive point estimate was not enough.

## Key lesson: predictive lift is not enough

One of the most important findings was a calibration failure mode:

| Calibration policy | Log loss | AUC | Brier | ECE10 |
|---|---:|---:|---:|---:|
| V8 Champion Strategy-D 45-55 | 0.594210 | 0.731161 | 0.205779 | 0.008098 |
| Wide gated isotonic 30-70 on Champion-only baseline | 0.614824 | 0.724022 | 0.211189 | 0.019681 |

The wide `0.30-0.70` isotonic gate looked attractive on validation, but failed badly on the locked test set. This is exactly why the project treats calibration policy selection as a risk-control problem, not just a metric-optimization exercise.

## Latest finding: V8-offset residual candidates

A later, cleaner experiment grafted the new features onto the **true V8 champion** as a
**no-intercept residual on the raw V8 logits** (rather than a fresh, weaker retrain), and
re-adjudicated every candidate under a **common out-of-fold beta calibrator** with paired-bootstrap
confidence intervals. Full table: [`reports/v8_offset_candidate_results.md`](reports/v8_offset_candidate_results.md).

- **`flash_csstats_pair`** — the strongest research candidate: **ΔAUC +0.00139**, **log-loss improvement: 0.00104**,
  ECE-neutral (P(ECE worsens) ≈ 0.40).
- **`csstats_pair`** — the most *reliable* backup: smaller gains but **P(improves) = 1.00** on both AUC
  and log loss.
- The edge is in **CSStats player-form pairs × flash**, not roster-overlap counts (those are
  neutral-to-negative on the V8 base).
- **Status:** `RC_candidate_for_forward_holdout` — small but reliable, **not** production-promoted.
- **Main blocker — coverage:** the CSStats features fire on only ~7% of test rows and coverage declines
  over time (train 17.6% → valid 11.5% → test 7.0%, spanning 2023-2026). The honest conclusion is a
  reliable but low-coverage **research candidate**, not a full historical Champion upgrade; raising
  CSStats join coverage is the biggest lever before any production promotion.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/cs2_pricing_research/
│   ├── metrics.py
│   ├── calibration.py
│   ├── evaluation.py          # bootstrap-CI eval, beta/temperature calibration,
│   │                          # per-map multicalibration, champion/challenger adjudication
│   └── validation_protocol.py
├── scripts/
│   ├── summarize_results.py
│   ├── calibration_policy_demo.py
│   └── validation_toolkit_demo.py   # end-to-end, synthetic, reproducible
├── reports/
│   ├── model_results.md
│   ├── v12_benchmark_results.csv
│   ├── v8_offset_candidate_results.md   # latest re-adjudication on the true V8 base
│   ├── v8_offset_candidate_results.csv
│   └── csstats_coverage_by_split.csv    # coverage limitation, verified from data
├── docs/
│   └── case_study.md
├── model_governance/
│   └── decision_log.md
└── sample_data/
    └── README.md
```

## Validation toolkit (`src/cs2_pricing_research/evaluation.py`)

The core of the project is not the model — it is the discipline for deciding whether a
change is a *real* improvement safe to release. The toolkit provides:

- **Bootstrap-CI evaluation** — paired bootstrap confidence intervals and `P(improves)`
  for log loss, Brier, AUC, and ECE (equal-width *and* adaptive/equal-frequency), so a
  candidate's metric move can be separated from sampling noise.
- **Full-range calibration** — temperature scaling and beta calibration, fit out-of-fold
  so there is no leakage; more stable across challengers than wide-band isotonic.
- **Per-map multicalibration** — group-wise reliability and calibration, because a low
  *marginal* ECE can hide real per-map miscalibration.
- **Champion/challenger adjudication** — a promotion decision made on a CI rule
  (proper scores must improve with high probability; ECE must not *reliably* worsen)
  rather than on a single point estimate.

A representative finding from the research: several feature candidates had been rejected
on ECE10 moves that were *inside the bootstrap noise band*, and some headline "wins" were
within noise and flipped sign under a neighbouring calibration policy — motivating the
CI-based ruler above.

## Methodology highlights

- Strict train -> valid -> locked test separation.
- Calibration fitted only on validation predictions.
- Log loss and ECE10 treated as primary metrics; AUC used for rank-order signal.
- One-at-a-time feature ablations before bundle testing.
- Candidate rejection or parking logged when calibration/reliability risk outweighed predictive lift.
- Champion artifacts were not mutated during research experiments.
- Forward validation required before any production-style promotion.

## Feature families investigated

- Map/team historical deltas.
- Flash execution and flash vulnerability signals.
- Round-micro tactical features.
- CSStats player recent-form features.
- Roster stability and synergy proxies.
- Calibration gates and residual/shrinkage policies.

## How to run the small demo

Install dependencies:

```bash
pip install -r requirements.txt
```

Print the included benchmark summary:

```bash
python scripts/summarize_results.py
```

Run a synthetic calibration-policy demo:

```bash
python scripts/calibration_policy_demo.py
```

Run the full validation-toolkit demo (bootstrap-CI evaluation, champion/challenger
adjudication, out-of-fold beta calibration, and per-map ECE) on synthetic data:

```bash
python scripts/validation_toolkit_demo.py
```

These demos do not reproduce the private research dataset. They demonstrate the reusable metric, calibration, and evaluation code patterns used in the project.

## Limitations

- Raw match, odds, and scraped data are not included.
- The included metrics are summarized research outputs.
- V12 rows are historical research diagnostics, not production promotion evidence.
- The historical test set has been used during research diagnostics, so promotion would require a fresh forward holdout.
- CSStats and roster/synergy features were `parked_for_review_not_promoted` until stronger bootstrap/forward evidence is available.

## Status

This repo is suitable as a technical portfolio/case-study artifact. It is not a production trading system.
