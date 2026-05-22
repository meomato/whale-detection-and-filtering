"""Split benchmark windows into train, validation, and test CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def prepare(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(args.windows)
    splits = pd.read_csv(args.splits)
    split_map = dict(zip(splits["filename"].astype(str), splits["split"].astype(str)))
    windows["split"] = windows["filename"].astype(str).map(split_map)
    if windows["split"].isna().any():
        missing = sorted(windows.loc[windows["split"].isna(), "filename"].astype(str).unique())
        raise ValueError(f"Missing split rows for {len(missing)} files")

    counts = {}
    for split in ("train", "val", "test"):
        part = windows[windows["split"].eq(split)].copy()
        part.to_csv(args.output_dir / f"{split}_windows.csv", index=False)
        counts[split] = {
            "rows": int(len(part)),
            "sound": int(part["label"].eq("sound").sum()),
            "noise": int(part["label"].eq("noise").sum()),
        }

    (args.output_dir / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
