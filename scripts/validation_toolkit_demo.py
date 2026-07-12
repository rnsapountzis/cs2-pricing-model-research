"""Reproducible, data-free demo of the calibration/validation toolkit.

Generates synthetic CS2-like map-winner predictions (predictions concentrated
near 0.5 with light tail miscalibration and per-map variation), then runs the
full "ruler":

  1. bootstrap-CI ECE            -> is a metric move real or sampling noise?
  2. paired re-adjudication      -> champion vs challenger on a CI rule
  3. full-range beta calibration -> out-of-fold, vs a naive baseline
  4. per-map ECE                 -> marginal calibration can hide group miscalibration

No private data is used; everything here is synthetic and deterministic.

Run:  python scripts/validation_toolkit_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs2_pricing_research.evaluation import (  # noqa: E402
    adjudicate,
    bootstrap_metric_ci,
    cross_fit_calibrate,
    paired_bootstrap_delta,
    per_group_ece,
    proper_scores,
)

RNG = np.random.default_rng(20260710)
MAPS = ["dust2", "mirage", "nuke", "ancient", "inferno", "overpass", "anubis"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_synthetic(n=3200):
    """A champion and a slightly-better challenger, with per-map miscalibration."""
    maps = RNG.choice(MAPS, size=n)
    true_logit = RNG.normal(0, 0.9, n)               # latent team-strength gap
    # per-map calibration offset the aggregate metric will partly hide
    map_bias = {m: b for m, b in zip(MAPS, RNG.normal(0, 0.25, len(MAPS)))}
    y = (RNG.uniform(size=n) < _sigmoid(true_logit + np.array([map_bias[m] for m in maps]))).astype(int)

    # champion: decent ranking, mild overconfidence in the tails
    champ = _sigmoid(0.85 * true_logit + RNG.normal(0, 0.35, n))
    # challenger: adds a weak-but-real signal (better ranking), similar calibration
    chal = _sigmoid(0.85 * true_logit + 0.20 * RNG.normal(0, 1, n) * 0 + 0.12 * true_logit
                    + RNG.normal(0, 0.34, n))
    return y, np.clip(champ, 1e-6, 1 - 1e-6), np.clip(chal, 1e-6, 1 - 1e-6), maps


def main():
    y, champ, chal, maps = make_synthetic()

    print("=" * 78)
    print("1) BOOTSTRAP-CI ECE  — the size of a 'real vs noise' move")
    print("=" * 78)
    ci = bootstrap_metric_ci(y, champ, "ece10", n_boot=1500, seed=1)
    print(f"champion ECE10 = {ci['point']:.4f}   95% CI [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]"
          f"   SE = {ci['se']:.4f}")
    print("Any challenger ECE delta smaller than ~this SE is not a calibration failure.\n")

    print("=" * 78)
    print("2) PAIRED RE-ADJUDICATION  — champion vs challenger on a CI rule")
    print("=" * 78)
    verdict = adjudicate(y, champ, chal, n_boot=1500, seed=2)
    print(verdict.table.to_string(index=False))
    for r in verdict.reasons:
        print("  -", r)
    print("  DECISION:", verdict.decision, "\n")

    print("=" * 78)
    print("3) FULL-RANGE BETA CALIBRATION  (out-of-fold) vs raw")
    print("=" * 78)
    beta = cross_fit_calibrate(champ, y, method="beta", n_folds=5, seed=3)
    for name, p in [("raw champion", champ), ("beta-calibrated", beta)]:
        s = proper_scores(y, p)
        print(f"  {name:16s} log_loss={s['log_loss']:.4f}  brier={s['brier']:.4f}  "
              f"ece10={s['ece10']:.4f}  ece10_adaptive={s['ece10_adaptive']:.4f}")
    print()

    print("=" * 78)
    print("4) PER-MAP ECE  — marginal calibration can hide group miscalibration")
    print("=" * 78)
    print(per_group_ece(y, champ, maps, min_n=30).to_string(index=False))


if __name__ == "__main__":
    main()
