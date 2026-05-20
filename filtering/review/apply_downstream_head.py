"""Apply a saved benchmark head to long-file embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score


def _segment_index_from_filename(filename: str) -> int | None:
    prefix = str(filename).split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def _add_global_times_from_filename(table: pd.DataFrame, segment_duration_s: float) -> pd.DataFrame:
    out = table.copy()
    segment_index = out["filename"].map(_segment_index_from_filename)
    if segment_index.notna().all():
        offset = (segment_index.astype(int) - 1) * float(segment_duration_s)
        out["global_start_s"] = offset + out["start_s"].astype(float)
        out["global_end_s"] = offset + out["end_s"].astype(float)
    return out


def _sound_overlap_s(start: float, end: float, events: pd.DataFrame) -> float:
    if events.empty:
        return 0.0
    starts = events["global_start_s"].astype(float).to_numpy()
    ends = events["global_end_s"].astype(float).to_numpy()
    overlap = np.maximum(0.0, np.minimum(end, ends) - np.maximum(start, starts))
    return float(overlap.sum())


def _add_labels_from_events(table: pd.DataFrame, events_path: Path, min_sound_overlap_s: float) -> pd.DataFrame:
    out = table.copy()
    events = pd.read_csv(events_path)
    labels = []
    names = []
    for row in out.itertuples(index=False):
        overlap = _sound_overlap_s(float(row.global_start_s), float(row.global_end_s), events)
        label = int(overlap >= float(min_sound_overlap_s))
        labels.append(label)
        names.append("sound" if label else "noise")
    out["label"] = labels
    out["label_name"] = names
    return out


def _score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def _merge_events(predictions: pd.DataFrame, threshold: float, merge_gap_s: float) -> pd.DataFrame:
    events: list[dict[str, float | str]] = []
    current: dict[str, float | str] | None = None
    positives = predictions.loc[predictions["score_sound"].astype(float) >= threshold].copy()
    positives = positives.sort_values(["global_start_s", "global_end_s"])
    for row in positives.itertuples(index=False):
        start = float(row.global_start_s)
        end = float(row.global_end_s)
        score = float(row.score_sound)
        if current is None or start > float(current["global_end_s"]) + merge_gap_s:
            if current is not None:
                events.append(current)
            current = {
                "recording_id": "long_file_A",
                "global_start_s": start,
                "global_end_s": end,
                "max_score": score,
                "mean_score": score,
                "window_count": 1,
            }
        else:
            count = int(current["window_count"])
            current["global_end_s"] = max(float(current["global_end_s"]), end)
            current["max_score"] = max(float(current["max_score"]), score)
            current["mean_score"] = (float(current["mean_score"]) * count + score) / (count + 1)
            current["window_count"] = count + 1
    if current is not None:
        events.append(current)
    out = pd.DataFrame(events)
    if not out.empty:
        out["duration_s"] = out["global_end_s"].astype(float) - out["global_start_s"].astype(float)
    return out


def apply_head(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X = np.load(args.embeddings)
    manifest = pd.read_csv(args.manifest)
    windows = pd.read_csv(args.windows)
    bundle = joblib.load(args.head)
    scaler = bundle["scaler"]
    model = bundle["model"]
    X_scaled = scaler.transform(X)
    scores = _score(model, X_scaled)
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
    table["pred_label"] = np.where(pred == 1, "sound", "noise")
    table["pred"] = pred
    table.to_csv(args.output_dir / "window_predictions.csv", index=False)

    events = _merge_events(table, float(args.threshold), float(args.merge_gap_s))
    events.to_csv(args.output_dir / "predicted_sound_events.csv", index=False)

    metrics: dict[str, float | int | str | None] = {
        "model_name": args.model_name,
        "head": str(args.head),
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
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--merge-gap-s", type=float, default=0.0)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--min-sound-overlap-s", type=float, default=0.25)
    parser.add_argument("--segment-duration-s", type=float, default=60.0)
    return parser.parse_args()


if __name__ == "__main__":
    apply_head(parse_args())
