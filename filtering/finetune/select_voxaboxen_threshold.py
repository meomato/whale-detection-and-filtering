"""Select a Voxaboxen detection threshold on validation and report test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_sweep(path: Path, iou: float) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    key = f"mAP@{iou:g}"
    if key not in data:
        key = f"mAP@{iou}"
    rows = data[key]["whale_sound"]
    return pd.DataFrame(rows).sort_values("det_thresh").reset_index(drop=True)


def _pick_threshold(sweep: pd.DataFrame) -> pd.Series:
    return (
        sweep.sort_values(["f1", "recall", "precision"], ascending=False)
        .reset_index(drop=True)
        .iloc[0]
    )


def _nearest_threshold_row(sweep: pd.DataFrame, threshold: float) -> pd.Series:
    idx = (sweep["det_thresh"].astype(float) - float(threshold)).abs().idxmin()
    return sweep.loc[idx]


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_sweep = _load_sweep(args.val_full_results, args.iou)
    test_sweep = _load_sweep(args.test_full_results, args.iou)
    selected = _pick_threshold(val_sweep)
    test = _nearest_threshold_row(test_sweep, float(selected["det_thresh"]))

    val_sweep.to_csv(args.output_dir / f"voxaboxen_val_threshold_sweep_iou_{args.iou:g}.csv", index=False)
    test_sweep.to_csv(args.output_dir / f"voxaboxen_test_threshold_sweep_iou_{args.iou:g}.csv", index=False)
    pd.DataFrame([selected]).to_csv(args.output_dir / "voxaboxen_selected_threshold_val.csv", index=False)
    pd.DataFrame([test]).to_csv(args.output_dir / "voxaboxen_test_at_val_threshold.csv", index=False)

    summary = {
        "model": "voxaboxen_beats",
        "threshold_selected_on": f"val_event_f1_iou_{args.iou:g}",
        "selected_threshold": float(selected["det_thresh"]),
        "val": {k: float(v) for k, v in selected.items() if isinstance(v, (int, float))},
        "test": {k: float(v) for k, v in test.items() if isinstance(v, (int, float))},
    }
    (args.output_dir / "voxaboxen_val_selected_threshold_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-full-results",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--test-full-results",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
