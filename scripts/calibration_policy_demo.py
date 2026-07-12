from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs2_pricing_research.calibration import apply_strategy_d, fit_gated_calibrators
from cs2_pricing_research.metrics import binary_classification_metrics


def main() -> None:
    rng = np.random.default_rng(42)
    n_train_like_valid = 3000
    n_test = 2000

    # Synthetic example: a model that has ranking signal but is over-confident in the mid-zone.
    latent_valid = rng.normal(size=n_train_like_valid)
    p_valid_raw = 1.0 / (1.0 + np.exp(-1.15 * latent_valid))
    true_p_valid = p_valid_raw.copy()
    valid_mid = (p_valid_raw >= 0.45) & (p_valid_raw <= 0.55)
    true_p_valid[valid_mid] = 0.5 + 0.35 * (p_valid_raw[valid_mid] - 0.5)
    y_valid = rng.binomial(1, true_p_valid)

    latent_test = rng.normal(size=n_test)
    p_test_raw = 1.0 / (1.0 + np.exp(-1.15 * latent_test))
    true_p_test = p_test_raw.copy()
    test_mid = (p_test_raw >= 0.45) & (p_test_raw <= 0.55)
    true_p_test[test_mid] = 0.5 + 0.35 * (p_test_raw[test_mid] - 0.5)
    y_test = rng.binomial(1, true_p_test)

    calibrators = fit_gated_calibrators(p_valid_raw, y_valid, gate_lo=0.45, gate_hi=0.55)
    p_test_strategy_d = apply_strategy_d(p_test_raw, calibrators, delta_threshold=0.035)

    print("Raw synthetic model:")
    print(binary_classification_metrics(y_test, p_test_raw))
    print("\nStrategy-D calibrated synthetic model:")
    print(binary_classification_metrics(y_test, p_test_strategy_d))


if __name__ == "__main__":
    main()
