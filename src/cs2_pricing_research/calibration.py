from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .metrics import clip01


def logit(p: np.ndarray | list[float]) -> np.ndarray:
    p2 = clip01(p)
    return np.log(p2 / (1.0 - p2))


@dataclass
class GatedCalibrators:
    gate_lo: float
    gate_hi: float
    gate_valid_rows: int
    isotonic: IsotonicRegression
    platt: LogisticRegression
    platt_coef: float
    platt_intercept: float


def fit_gated_calibrators(
    p_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    gate_lo: float = 0.45,
    gate_hi: float = 0.55,
    min_gate_rows: int = 100,
) -> GatedCalibrators:
    """Fit Champion-style gated isotonic and Platt calibrators on validation predictions only."""
    p = clip01(p_valid)
    y = np.asarray(y_valid, dtype=int)
    gate = (p >= gate_lo) & (p <= gate_hi)
    gate_rows = int(gate.sum())
    if gate_rows < min_gate_rows or len(np.unique(y[gate])) < 2:
        raise ValueError(f"Not enough valid gate rows: gate_rows={gate_rows}")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(p[gate], y[gate])

    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=23)
    platt.fit(logit(p[gate]).reshape(-1, 1), y[gate])

    return GatedCalibrators(
        gate_lo=float(gate_lo),
        gate_hi=float(gate_hi),
        gate_valid_rows=gate_rows,
        isotonic=iso,
        platt=platt,
        platt_coef=float(platt.coef_[0][0]),
        platt_intercept=float(platt.intercept_[0]),
    )


def apply_gated_isotonic(p_raw: np.ndarray, calibrators: GatedCalibrators) -> np.ndarray:
    p = clip01(p_raw)
    gate = (p >= calibrators.gate_lo) & (p <= calibrators.gate_hi)
    out = p.copy()
    if int(gate.sum()):
        out[gate] = clip01(calibrators.isotonic.predict(p[gate]))
    return clip01(out)


def apply_gated_platt(p_raw: np.ndarray, calibrators: GatedCalibrators) -> np.ndarray:
    p = clip01(p_raw)
    gate = (p >= calibrators.gate_lo) & (p <= calibrators.gate_hi)
    out = p.copy()
    if int(gate.sum()):
        out[gate] = clip01(calibrators.platt.predict_proba(logit(p[gate]).reshape(-1, 1))[:, 1])
    return clip01(out)


def apply_strategy_d(
    p_raw: np.ndarray,
    calibrators: GatedCalibrators,
    *,
    delta_threshold: float = 0.035,
) -> np.ndarray:
    """Strategy-D: use gated isotonic unless the jump is too large, then fall back to Platt."""
    p = clip01(p_raw)
    p_iso = apply_gated_isotonic(p, calibrators)
    p_platt = apply_gated_platt(p, calibrators)
    gate = (p >= calibrators.gate_lo) & (p <= calibrators.gate_hi)
    fallback = gate & (np.abs(p_iso - p) > delta_threshold)
    out = p_iso.copy()
    out[fallback] = p_platt[fallback]
    return clip01(out)
