"""Apply CV heads to long-file embeddings and average their scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from filtering.review.apply_downstream_head import (
    _add_global_times_from_filename,
    _add_labels_from_events,
    _merge_events,
    _score,
)


def _load_scores(heads: list[Path], x: np.ndarray) -> tuple[np.ndarray, list[dict[str, str | float]]]:
    fold_scores = []
    rows = []
    for head in heads:
        bundle = joblib.load(head)
        x_scaled = bundle["scaler"].transform(x)
        scores = _score(bundle["model"], x_scaled)
        fold_scores.append(scores)
        rows.append(
            {
                "head": str(head),
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
            }
        )
    return np.vstack(fold_scores), rows


def apply_ensemble(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heads = sorted(args.heads_dir.glob("fold_*/model.joblib"))
    if not heads:
        raise FileNotFoundError(f"No fold heads found under {args.heads_dir}")

    x = np.load(args.embeddings)
    manifest = pd.read_csv(args.manifest)
    windows = pd.read_csv(args.windows)
    fold_scores, head_rows = _load_scores(heads, x)
    scores = fold_scores.mean(axis=0)
    score_std = fold_scores.std(axis=0)
    pred = (scores >= float(args.threshold)).astype(int)

    table = manifest.copy()
    if "row_index" in table.columns and "row_index" in windows.columns:
        table = table.merge(
            windows[
                [
                    "filename",
                    "start_s",
                    "end_s",
                    "global_start_s",
                    "global_end_s",
                    "label",
                    "label_name",
                ]
            ],
            on=["filename", "start_s", "end_s"],
            how="left",
        )
    else:
        table["global_start_s"] = table["start_s"]
        table["global_end_s"] = table["end_s"]
    if "global_start_s" not in table.columns or table["global_start_s"].isna().any():
        table = _add_global_times_from_filename(table, float(args.segment_duration_s))
    if args.events and ("label" not in table.columns or table["label"].isna().any()):
        table = _add_labels_from_events(table, args.events, float(args.min_sound_overlap_s))

    table["score_sound"] = scores
    table["score_sound_std"] = score_std
    table["pred"] = pred
    table["pred_label"] = np.where(pred == 1, "sound", "noise")
    table.to_csv(args.output_dir / "window_predictions.csv", index=False)

    events = _merge_events(table, float(args.threshold), float(args.merge_gap_s))
    events.to_csv(args.output_dir / "predicted_sound_events.csv", index=False)
    pd.DataFrame(head_rows).to_csv(args.output_dir / "ensemble_heads.csv", index=False)

    metrics: dict[str, float | int | str | None] = {
        "model_name": args.model_name,
        "heads_dir": str(args.heads_dir),
        "n_heads": int(len(heads)),
        "embeddings": str(args.embeddings),
        "threshold": float(args.threshold),
        "merge_gap_s": float(args.merge_gap_s),
        "windows": int(len(table)),
        "predicted_sound_windows": int(pred.sum()),
        "predicted_sound_events": int(len(events)),
    }
    if "label" in table.columns and table["label"].notna().any() and table["label"].nunique() > 1:
        y_true = table["label"].astype(int).to_numpy()
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
        metrics.update(
            {
                "window_precision_sound": float(p[1]),
                "window_recall_sound": float(r[1]),
                "window_f1_sound": float(f1[1]),
                "window_macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
                "window_average_precision": float(average_precision_score(y_true, scores)),
                "window_roc_auc": float(roc_auc_score(y_true, scores)),
            }
        )
    (args.output_dir / "inference_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--heads-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--merge-gap-s", type=float, default=1.0)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--min-sound-overlap-s", type=float, default=0.25)
    parser.add_argument("--segment-duration-s", type=float, default=60.0)
    return parser.parse_args()


if __name__ == "__main__":
    apply_ensemble(parse_args())
