from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitCounts:
    train: int
    valid: int
    test: int


def split_counts(df: pd.DataFrame, split_col: str = "__split") -> SplitCounts:
    counts = df[split_col].value_counts().to_dict()
    return SplitCounts(
        train=int(counts.get("train", 0)),
        valid=int(counts.get("valid", 0)),
        test=int(counts.get("test", 0)),
    )


def assert_no_duplicate_keys(df: pd.DataFrame, keys: list[str]) -> None:
    dupes = int(df.duplicated(keys).sum())
    if dupes:
        raise ValueError(f"Found {dupes} duplicate rows for keys={keys}")


def feature_coverage(df: pd.DataFrame, features: list[str], split_col: str = "__split") -> pd.DataFrame:
    rows = []
    for split, g in df.groupby(split_col):
        rec = {"split": split, "rows": len(g)}
        for feature in features:
            rec[f"{feature}__nonnull"] = int(g[feature].notna().sum())
            rec[f"{feature}__coverage"] = float(g[feature].notna().mean())
        rows.append(rec)
    return pd.DataFrame(rows)

