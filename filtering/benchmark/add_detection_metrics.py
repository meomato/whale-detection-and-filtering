"""Add detection metrics to saved benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filtering.benchmark.train_downstream import _detection_metrics


def update_run(
    metrics_path: Path,
    decision_threshold: float,
    event_iou_threshold: float,
    event_merge_gap_s: float,
) -> bool:
    predictions_path = metrics_path.parent / "test_predictions.csv"
    if not predictions_path.is_file():
        return False

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    if "score_sound" not in predictions.columns or "label" not in predictions.columns:
        return False

    test = metrics.setdefault("test", {})
    if "average_precision" in test:
        test["map_sound"] = test["average_precision"]
    if {"label", "score_sound"}.issubset(predictions.columns) and predictions["label"].nunique() > 1:
        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(
            predictions["label"].to_numpy(),
            predictions["score_sound"].to_numpy(),
        )
        for target_precision in (0.8, 0.9):
            mask = precision >= target_precision
            test[f"recall_at_precision_{target_precision:.1f}"] = float(recall[mask].max()) if mask.any() else 0.0

    annotations = Path(str(metrics["annotations"]))
    metrics["detection"] = _detection_metrics(
        predictions,
        predictions["score_sound"].to_numpy(),
        annotations,
        event_iou_threshold,
        event_merge_gap_s,
        decision_threshold,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return True


def update_all(args: argparse.Namespace) -> None:
    updated = 0
    for metrics_path in sorted(args.runs_dir.glob("*/*/metrics.json")):
        if update_run(metrics_path, args.decision_threshold, args.event_iou_threshold, args.event_merge_gap_s):
            updated += 1
    print(f"Updated {updated} runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/benchmark/runs"))
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--event-iou-threshold", type=float, default=0.1)
    parser.add_argument("--event-merge-gap-s", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    update_all(parse_args())
