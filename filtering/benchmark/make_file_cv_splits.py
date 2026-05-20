"""Create file-level CV splits from a benchmark split table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _file_bins(windows: pd.DataFrame) -> pd.DataFrame:
    work = windows.copy()
    work["label_binary"] = work["label"].map({"noise": 0, "sound": 1}).fillna(work["label"]).astype(int)
    grouped = work.groupby("filename", as_index=False)["label_binary"].agg(["sum", "count"]).reset_index()
    grouped["sound_fraction"] = grouped["sum"] / grouped["count"].clip(lower=1)
    n_unique = grouped["sound_fraction"].nunique()
    n_bins = min(4, max(2, n_unique))
    try:
        grouped["bin"] = pd.qcut(grouped["sound_fraction"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        grouped["bin"] = (grouped["sound_fraction"] > 0).astype(int)
    grouped["bin"] = grouped["bin"].fillna(0).astype(int)
    return grouped


def make_splits(windows_path: Path, base_splits_path: Path, output_dir: Path, folds: int, seed: int) -> None:
    windows = pd.read_csv(windows_path)
    windows["label_binary"] = windows["label"].map({"noise": 0, "sound": 1}).fillna(windows["label"]).astype(int)
    base_splits = pd.read_csv(base_splits_path)
    files = _file_bins(windows)
    if len(files) < folds:
        raise ValueError(f"Need at least {folds} files, got {len(files)}")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    file_names = files["filename"].astype(str).to_numpy()
    bins = files["bin"].to_numpy()
    fold_indices = list(splitter.split(file_names, bins))

    for fold_idx, (_, test_idx) in enumerate(fold_indices, start=1):
        val_idx = fold_indices[fold_idx % folds][1]
        test_files = set(file_names[test_idx])
        val_files = set(file_names[val_idx])

        split_rows = []
        for row in base_splits.itertuples(index=False):
            filename = str(row.filename)
            if filename in test_files:
                split = "test"
            elif filename in val_files:
                split = "val"
            else:
                split = "train"
            split_rows.append(
                {
                    "filename": filename,
                    "split": split,
                    "source": getattr(row, "source", ""),
                    "item_id": getattr(row, "item_id", ""),
                }
            )

        fold_dir = output_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_table = pd.DataFrame(split_rows)
        split_table.to_csv(fold_dir / "splits.csv", index=False)

        counts = split_table["split"].value_counts().to_dict()
        for split_name, split_files in split_table.groupby("split"):
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "split": split_name,
                    "files": int(len(split_files)),
                    "windows": int(windows["filename"].isin(split_files["filename"]).sum()),
                    "sound_windows": int(
                    windows.loc[windows["filename"].isin(split_files["filename"]), "label_binary"].sum()
                    ),
                }
            )
        print(f"fold_{fold_idx:02d}: {counts}")

    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--base-splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_splits(args.windows, args.base_splits, args.output_dir, args.folds, args.seed)
