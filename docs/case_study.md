# Case Study: CS2 Map-Winner Pricing Research

## 1. Problem

The goal was to improve CS2 map-winner probability estimates while preserving calibration quality. In betting and trading workflows, better rank ordering is not enough: probabilities also need to be usable for pricing and risk decisions.

Primary metrics:

- **Log loss**: probability quality / entropy.
- **AUC**: rank-order discrimination.
- **Brier score**: squared probability error.
- **ECE10**: 10-bin expected calibration error.

## 2. Baseline

The baseline was a V8 Champion logistic-regression model using 71 raw features:

- map/team historical deltas;
- flash micro features;
- map-id controls;
- Champion-style Strategy-D gated calibration.

Baseline locked-test metrics:

| Model | Log loss | AUC | Brier | ECE10 |
|---|---:|---:|---:|---:|
| V8 Champion, Strategy-D 45-55 | 0.594210 | 0.731161 | 0.205779 | 0.008098 |

## 3. Feature research

Several CS2-specific feature families were tested.

### Flash execution

Features around flash impact, vulnerability, and execution quality showed repeatable signal. The cleanest V12 raw-retrain candidate was `v9_flash_execution_roi_delta`.

### CSStats player form

Player match-summary data was converted into team-vs-opponent deltas. CSStats features showed attractive point estimates in some tests, but they did not clear the project's bootstrap reliability bar when evaluated as separate recent-form candidates. They were therefore parked for review rather than promoted.

### Roster/synergy

Roster continuity and prior teammate history showed predictive signal, but some history-count features behaved like broad tier/coverage proxies and created calibration risk. They were not promoted.

## 4. Calibration puzzle

Many features improved log loss and AUC but worsened ECE10. The diagnosis was:

- The existing Champion output was already calibrated for the old feature space.
- Adding residual corrections on top of calibrated probabilities often pushed predictions too aggressively.
- A valid research experiment was to retrain the raw model with new features, then re-calibrate from scratch.
- A promotion decision still required bootstrap reliability and a fresh forward holdout.

This became the V12 raw-retrain/fresh-calibration lab.

## 5. V12 raw-retrain diagnostic

V12 used:

```text
Champion 71 raw features
    + selected new CS2 features
    -> raw logistic regression retrain
    -> fresh gated isotonic / Platt / Strategy-D calibration on validation
    -> locked test diagnostics
```

The most defensible single new feature was flash ROI:

| Candidate | Log loss | AUC | Brier | ECE10 | Delta log loss | Delta AUC | Delta ECE10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8 Champion | 0.594210 | 0.731161 | 0.205779 | 0.008098 | 0.000000 | 0.000000 | 0.000000 |
| V12 flash ROI | 0.593572 | 0.731271 | 0.205547 | 0.009837 | -0.000638 | +0.000109 | +0.001739 |

The lab also produced combined CSStats diagnostics:

| Candidate | Log loss | AUC | Brier | ECE10 | Status |
|---|---:|---:|---:|---:|---|
| V12 flash ROI + CSStats assists | 0.593345 | 0.731710 | 0.205451 | 0.011058 | `diagnostic_only_not_promoted` |
| V12 flash ROI + CSStats assists/rating30 | 0.592811 | 0.732768 | 0.205236 | 0.011931 | `diagnostic_only_ece_reliability_risk` |

Provenance and interpretation note:

- These combined CSStats rows are real V12 raw-retrain diagnostics from the private workspace.
- They are not promotion claims.
- They do not override the earlier CSStats bootstrap decision, where standalone recent-form candidates failed the reliability bar.
- CSStats therefore remained `parked_for_review_not_promoted` pending more coverage and forward validation.

## 6. Failure analysis

A wide validation-selected isotonic gate looked good on validation but failed badly on test:

| Policy | Log loss | AUC | Brier | ECE10 |
|---|---:|---:|---:|---:|
| Strategy-D 45-55 | 0.594210 | 0.731161 | 0.205779 | 0.008098 |
| Wide isotonic 30-70 | 0.614824 | 0.724022 | 0.211189 | 0.019681 |

This was treated as a calibration overfit incident. The lesson was not to widen the gate until validation looks good.

## 7. Governance reversal: CSStats recent form

An important governance moment was the CSStats recent-form reversal:

- Some CSStats candidates had attractive point estimates.
- Bootstrap reliability showed that no CSStats candidate cleared the project's Step-8-style evidence bar.
- A preliminary RC-style idea was explicitly pulled back.
- The features were classified as `parked_for_review_not_promoted`.

This is a core part of the case study: the process was designed to catch tempting but insufficiently reliable signals.

## 8. Governance

Research candidates were classified into:

- champion reference;
- research candidate;
- diagnostic only / parked for review, not promoted;
- predictive but ECE-risk;
- rejected due to calibration risk or data coverage risk.

No V12 candidate is claimed as production-promoted. The correct next step is a fresh forward holdout with pre-registered sample-size and metric gates.

## 9. What this demonstrates

This project demonstrates:

- CS2 domain-aware feature engineering;
- calibration-aware model validation;
- incident-style analysis of model failures;
- disciplined train/valid/test separation;
- practical thinking around production trading risk;
- ability to translate research results into trader-readable decisions.
