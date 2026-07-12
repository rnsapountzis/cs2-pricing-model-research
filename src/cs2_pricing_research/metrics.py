from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def clip01(p: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """Clip probabilities away from exact 0/1 for stable log loss/logit operations."""
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def ece_score(
    y_true: np.ndarray | pd.Series | list[int],
    p_pred: np.ndarray | pd.Series | list[float],
    *,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Expected calibration error with equal-width probability bins."""
    y = np.asarray(y_true, dtype=float)
    p = clip01(p_pred)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows: list[dict[str, float | int]] = []

    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "lo": lo,
                    "hi": hi,
                    "n": 0,
                    "avg_pred": np.nan,
                    "actual_rate": np.nan,
                    "gap": np.nan,
                    "weighted_abs_gap": 0.0,
                }
            )
            continue

        avg_pred = float(p[mask].mean())
        actual_rate = float(y[mask].mean())
        gap = actual_rate - avg_pred
        weighted_abs_gap = (n / len(y)) * abs(gap)
        ece += weighted_abs_gap
        rows.append(
            {
                "bin": i + 1,
                "lo": lo,
                "hi": hi,
                "n": n,
                "avg_pred": avg_pred,
                "actual_rate": actual_rate,
                "gap": gap,
                "weighted_abs_gap": weighted_abs_gap,
            }
        )

    return float(ece), pd.DataFrame(rows)


def binary_classification_metrics(
    y_true: np.ndarray | pd.Series | list[int],
    p_pred: np.ndarray | pd.Series | list[float],
) -> dict[str, float | int]:
    """Common pricing-model metrics for binary map-winner probabilities."""
    y = np.asarray(y_true, dtype=int)
    p = clip01(p_pred)
    ece10, _ = ece_score(y, p, n_bins=10)
    return {
        "rows": int(len(y)),
        "base_rate": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
        "pred_std": float(np.std(p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "ece10": float(ece10),
    }

