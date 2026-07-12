"""
CS2 calibration toolkit — drop-in extensions for the pricing-model research repo.

Design goals (match existing src/cs2_pricing_research style):
- pure numpy/pandas/sklearn, no side effects on import;
- every public function is deterministic given a seed;
- nothing here fits on TEST — calibration fitting is valid-only or out-of-fold.

Modules:
  1. Bootstrap-CI evaluation  -> paired bootstrap CIs + P(improve) for any metric,
                                 incl. equal-width AND adaptive (equal-frequency) ECE.
  2. Full-range calibration   -> temperature scaling + beta calibration, with
                                 out-of-fold cross-fitting so there is no leakage.
  3. Per-map / group ECE       -> per-group reliability table + group-wise calibration
                                 (multicalibration) with a min-n fallback to global.
  4. Re-adjudication harness   -> champion vs challenger decided on a CI rule, not a
                                 point estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
EPS = 1e-6


def clip01(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p) -> np.ndarray:
    p2 = clip01(p)
    return np.log(p2 / (1.0 - p2))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# --------------------------------------------------------------------------- #
# 1. metrics: equal-width ECE (matches repo), adaptive ECE, proper scores
# --------------------------------------------------------------------------- #
def ece_equal_width(y_true, p_pred, *, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins. Matches metrics.ece_score."""
    y = np.asarray(y_true, dtype=float)
    p = clip01(p_pred)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / len(y)) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def ece_adaptive(y_true, p_pred, *, n_bins: int = 10) -> float:
    """Adaptive ECE with equal-FREQUENCY bins (quantile edges).

    Far less sharpness-confounded than equal-width ECE when predictions cluster
    near 0.5 (as CS2 map probabilities do), because every bin carries ~equal mass.
    """
    y = np.asarray(y_true, dtype=float)
    p = clip01(p_pred)
    order = np.argsort(p)
    y, p = y[order], p[order]
    ece = 0.0
    idx = np.array_split(np.arange(len(y)), n_bins)
    for chunk in idx:
        if len(chunk) == 0:
            continue
        ece += (len(chunk) / len(y)) * abs(y[chunk].mean() - p[chunk].mean())
    return float(ece)


def proper_scores(y_true, p_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = clip01(p_pred)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "ece10": ece_equal_width(y, p, n_bins=10),
        "ece10_adaptive": ece_adaptive(y, p, n_bins=10),
    }


# metric sign convention: +1 => higher is better, -1 => lower is better
METRIC_DIRECTION = {
    "auc": +1,
    "log_loss": -1,
    "brier": -1,
    "ece10": -1,
    "ece10_adaptive": -1,
}


# --------------------------------------------------------------------------- #
# 1b. bootstrap CIs — the new "ruler"
# --------------------------------------------------------------------------- #
@dataclass
class DeltaCI:
    metric: str
    point_champion: float
    point_challenger: float
    delta_mean: float           # challenger - champion, oriented so + = better
    p025: float
    p50: float
    p975: float
    prob_improves: float        # P(challenger better than champion) over resamples

    def as_row(self) -> dict:
        return {
            "metric": self.metric,
            "champion": round(self.point_champion, 6),
            "challenger": round(self.point_challenger, 6),
            "delta_better_is_pos": round(self.delta_mean, 6),
            "ci_lo": round(self.p025, 6),
            "ci_hi": round(self.p975, 6),
            "prob_improves": round(self.prob_improves, 3),
        }


def paired_bootstrap_delta(
    y_true,
    p_champion,
    p_challenger,
    *,
    metrics: tuple[str, ...] = ("log_loss", "brier", "auc", "ece10", "ece10_adaptive"),
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, DeltaCI]:
    """Paired bootstrap: resample ROWS once per iteration and score BOTH models on
    the same resampled rows. Removes shared-sample noise, so tiny deltas are
    measured against their real sampling variability.

    delta is oriented so that positive always means "challenger is better".
    """
    y = np.asarray(y_true, dtype=int)
    pc = clip01(p_champion)
    ph = clip01(p_challenger)
    n = len(y)
    rng = np.random.default_rng(seed)

    def score(metric, yy, pp):
        if metric == "log_loss":
            return log_loss(yy, pp, labels=[0, 1])
        if metric == "brier":
            return brier_score_loss(yy, pp)
        if metric == "auc":
            return roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan
        if metric == "ece10":
            return ece_equal_width(yy, pp, n_bins=10)
        if metric == "ece10_adaptive":
            return ece_adaptive(yy, pp, n_bins=10)
        raise KeyError(metric)

    point = {m: (score(m, y, pc), score(m, y, ph)) for m in metrics}
    boot = {m: np.empty(n_boot) for m in metrics}

    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            # degenerate resample; reuse previous to avoid nan contamination
            for m in metrics:
                boot[m][b] = boot[m][b - 1] if b else 0.0
            continue
        cc, hh = pc[idx], ph[idx]
        for m in metrics:
            d = score(m, yy, hh) - score(m, yy, cc)      # challenger - champion
            boot[m][b] = d * METRIC_DIRECTION[m]         # orient: + = better
    out = {}
    for m in metrics:
        arr = boot[m]
        pc_pt, ph_pt = point[m]
        out[m] = DeltaCI(
            metric=m,
            point_champion=pc_pt,
            point_challenger=ph_pt,
            delta_mean=float(np.mean(arr)),
            p025=float(np.percentile(arr, 2.5)),
            p50=float(np.percentile(arr, 50)),
            p975=float(np.percentile(arr, 97.5)),
            prob_improves=float(np.mean(arr > 0)),
        )
    return out


def bootstrap_metric_ci(
    y_true, p_pred, metric: str = "ece10", *, n_boot: int = 2000, seed: int = 42
) -> dict[str, float]:
    """Absolute-level bootstrap CI for a single model's metric (e.g. ECE10 ± CI)."""
    y = np.asarray(y_true, dtype=int)
    p = clip01(p_pred)
    n = len(y)
    rng = np.random.default_rng(seed)
    fn = {
        "ece10": lambda yy, pp: ece_equal_width(yy, pp, n_bins=10),
        "ece10_adaptive": lambda yy, pp: ece_adaptive(yy, pp, n_bins=10),
        "log_loss": lambda yy, pp: log_loss(yy, pp, labels=[0, 1]),
        "brier": lambda yy, pp: brier_score_loss(yy, pp),
        "auc": lambda yy, pp: roc_auc_score(yy, pp),
    }[metric]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        vals[b] = fn(yy, pp) if len(np.unique(yy)) == 2 else np.nan
    vals = vals[~np.isnan(vals)]
    return {
        "point": float(fn(y, p)),
        "mean": float(vals.mean()),
        "ci_lo": float(np.percentile(vals, 2.5)),
        "ci_hi": float(np.percentile(vals, 97.5)),
        "se": float(vals.std(ddof=1)),
    }


# --------------------------------------------------------------------------- #
# 2. full-range calibration: temperature + beta, with out-of-fold cross-fit
# --------------------------------------------------------------------------- #
@dataclass
class TemperatureCalibrator:
    T: float

    def predict(self, p_raw) -> np.ndarray:
        return clip01(sigmoid(logit(p_raw) / self.T))


def fit_temperature(p_raw, y, *, bounds=(0.2, 5.0)) -> TemperatureCalibrator:
    z = logit(p_raw)
    y = np.asarray(y, dtype=int)

    def nll(T):
        p = clip01(sigmoid(z / T))
        return log_loss(y, p, labels=[0, 1])

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return TemperatureCalibrator(T=float(res.x))


@dataclass
class BetaCalibrator:
    """Beta calibration (Kull et al. 2017): logistic fit on [ln p, ln(1-p)].

    Full-range, monotone in practice, only 3 parameters -> stable across
    challengers, unlike wide-band isotonic which overfits.
    """
    coef_a: float
    coef_b: float
    intercept: float

    def predict(self, p_raw) -> np.ndarray:
        p = clip01(p_raw)
        z = self.coef_a * np.log(p) + self.coef_b * np.log(1.0 - p) + self.intercept
        return clip01(sigmoid(z))


def fit_beta(p_raw, y) -> BetaCalibrator:
    p = clip01(p_raw)
    X = np.column_stack([np.log(p), np.log(1.0 - p)])
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    lr.fit(X, np.asarray(y, dtype=int))
    return BetaCalibrator(
        coef_a=float(lr.coef_[0][0]),
        coef_b=float(lr.coef_[0][1]),
        intercept=float(lr.intercept_[0]),
    )


def cross_fit_calibrate(
    p_raw, y, *, method: str = "beta", n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Out-of-fold calibrated predictions: fit on k-1 folds, predict held-out fold.

    Use this to evaluate a calibration method honestly when you don't have a
    separate valid split (no row is ever calibrated by a model fit on itself).
    In production you would instead fit ONE calibrator on the valid split.
    """
    p_raw = clip01(p_raw)
    y = np.asarray(y, dtype=int)
    out = np.empty_like(p_raw)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fitter = {"beta": fit_beta, "temperature": fit_temperature}[method]
    for tr, te in skf.split(p_raw, y):
        cal = fitter(p_raw[tr], y[tr])
        out[te] = cal.predict(p_raw[te])
    return out


# --------------------------------------------------------------------------- #
# 3. per-map / group ECE + group-wise (multi)calibration
# --------------------------------------------------------------------------- #
def per_group_ece(
    y_true, p_pred, groups, *, n_bins: int = 10, min_n: int = 30
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = clip01(p_pred)
    g = np.asarray(groups)
    rows = []
    for gv in pd.unique(g):
        m = g == gv
        n = int(m.sum())
        rows.append({
            "group": gv,
            "n": n,
            "base_rate": float(y[m].mean()) if n else np.nan,
            "pred_mean": float(p[m].mean()) if n else np.nan,
            "ece10": ece_equal_width(y[m], p[m], n_bins=n_bins) if n >= min_n else np.nan,
            "ece10_adaptive": ece_adaptive(y[m], p[m], n_bins=n_bins) if n >= min_n else np.nan,
            "log_loss": log_loss(y[m], p[m], labels=[0, 1]) if n >= min_n else np.nan,
            "reliable": n >= min_n,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def cross_fit_group_calibrate(
    p_raw, y, groups, *, method: str = "beta", n_folds: int = 5,
    min_group_n: int = 150, seed: int = 42,
) -> np.ndarray:
    """Multicalibration by group (e.g. per map): out-of-fold group-wise calibrator,
    falling back to a global calibrator for groups with too few rows to fit safely.
    """
    p_raw = clip01(p_raw)
    y = np.asarray(y, dtype=int)
    g = np.asarray(groups)
    out = np.empty_like(p_raw)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fitter = {"beta": fit_beta, "temperature": fit_temperature}[method]
    for tr, te in skf.split(p_raw, y):
        global_cal = fitter(p_raw[tr], y[tr])
        cals = {}
        for gv in np.unique(g[tr]):
            mtr = tr[g[tr] == gv]
            if len(mtr) >= min_group_n and len(np.unique(y[mtr])) == 2:
                cals[gv] = fitter(p_raw[mtr], y[mtr])
        for i in te:
            cal = cals.get(g[i], global_cal)
            out[i] = cal.predict(p_raw[i : i + 1])[0]
    return out


# --------------------------------------------------------------------------- #
# 4. re-adjudication harness
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    table: pd.DataFrame
    decision: str
    reasons: list[str] = field(default_factory=list)


def adjudicate(
    y_true,
    p_champion,
    p_challenger,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    proper_prob_gate: float = 0.90,
    ece_worsen_gate: float = 0.80,
) -> Verdict:
    """Decide champion-vs-challenger on bootstrap CIs.

    Promote if:
      P(log_loss improves) >= gate AND P(brier improves) >= gate
      AND ECE10 does not RELIABLY worsen  (P(ece10 worsens) < ece_worsen_gate,
                                            i.e. CI still includes 0).
    """
    d = paired_bootstrap_delta(
        y_true, p_champion, p_challenger, n_boot=n_boot, seed=seed
    )
    table = pd.DataFrame([d[m].as_row() for m in d])
    p_ll = d["log_loss"].prob_improves
    p_br = d["brier"].prob_improves
    p_ece_worsens = 1.0 - d["ece10"].prob_improves
    p_ece_adapt_worsens = 1.0 - d["ece10_adaptive"].prob_improves

    reasons = []
    proper_ok = p_ll >= proper_prob_gate and p_br >= proper_prob_gate
    reasons.append(
        f"proper scores: P(logloss↑)={p_ll:.2f}, P(brier↑)={p_br:.2f} "
        f"-> {'PASS' if proper_ok else 'FAIL'} (gate {proper_prob_gate})"
    )
    ece_ok = p_ece_worsens < ece_worsen_gate
    reasons.append(
        f"calibration guardrail: P(ECE10 worsens)={p_ece_worsens:.2f} "
        f"(adaptive {p_ece_adapt_worsens:.2f}) -> "
        f"{'PASS' if ece_ok else 'FAIL'} (must be < {ece_worsen_gate})"
    )
    if proper_ok and ece_ok:
        decision = "PROMOTE_TO_FORWARD_HOLDOUT"
    elif proper_ok and not ece_ok:
        decision = "PARK_RELIABLE_DECALIBRATION"
    else:
        decision = "REJECT_NO_PROPER_GAIN"
    return Verdict(table=table, decision=decision, reasons=reasons)
