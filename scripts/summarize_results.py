from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "v12_benchmark_results.csv"


def main() -> None:
    df = pd.read_csv(RESULTS)
    baseline = df[df["feature_set"].eq("v8_champion_strategy_d_45_55")].iloc[0]
    display = df.copy()
    display["delta_log_loss"] = display["log_loss"] - baseline["log_loss"]
    display["delta_auc"] = display["auc"] - baseline["auc"]
    display["delta_ece10"] = display["ece10"] - baseline["ece10"]
    cols = ["feature_set", "log_loss", "auc", "brier", "ece10", "delta_log_loss", "delta_auc", "delta_ece10", "status"]
    print(display[cols].to_string(index=False))


if __name__ == "__main__":
    main()

