"""Apply HPC CV embedding heads to long-file embeddings."""

from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score

from filtering.finetune.train_embedding_head_detector import BinaryHead
from filtering.review.apply_downstream_head import _add_global_times_from_filename, _add_labels_from_events, _merge_events


def _torch_load_hpc(path: Path) -> dict:
    original_posix = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original_posix


def _load_threshold(heads_dir: Path, default: float) -> float:
    values: list[float] = []
    for path in sorted(heads_dir.glob("fold_*/selected_threshold.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        values.append(float(data["selected_threshold"]))
    return float(np.mean(values)) if values else float(default)


def _load_scores(heads_dir: Path, x: np.ndarray) -> tuple[np.ndarray, list[dict[str, float | str | int]]]:
    fold_scores = []
    rows = []
    for checkpoint_path in sorted(heads_dir.glob("fold_*/best_head.pt")):
        checkpoint = _torch_load_hpc(checkpoint_path)
        args = checkpoint["args"]
        scaler = joblib.load(checkpoint_path.parent / "scaler.joblib")["scaler"]
        x_scaled = scaler.transform(x).astype(np.float32)
        model = BinaryHead(
            input_dim=x_scaled.shape[1],
            head_kind=str(args["head_kind"]),
            hidden_dim=int(args["hidden_dim"]),
            dropout=float(args["dropout"]),
        )
        model.load_state_dict(checkpoint["head"])
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.from_numpy(x_scaled))).cpu().numpy()
        fold_scores.append(scores)
        rows.append(
            {
                "checkpoint": str(checkpoint_path),
                "epoch": int(checkpoint.get("epoch", 0)),
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
            }
        )
    if not fold_scores:
        raise FileNotFoundError(f"No fold_*/best_head.pt checkpoints under {heads_dir}")
    return np.vstack(fold_scores), rows


def apply_heads(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = np.load(args.embeddings).astype(np.float32)
    manifest = pd.read_csv(args.manifest)
    windows = pd.read_csv(args.windows)
    fold_scores, fold_rows = _load_scores(args.heads_dir, x)
    scores = fold_scores.mean(axis=0)
    score_std = fold_scores.std(axis=0)
    threshold = _load_threshold(args.heads_dir, args.threshold)
    pred = (scores >= threshold).astype(int)

    table = manifest.copy()
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
    if "global_start_s" not in table.columns or table["global_start_s"].isna().any():
        table = _add_global_times_from_filename(table, float(args.segment_duration_s))
    if args.events and ("label" not in table.columns or table["label"].isna().any()):
        table = _add_labels_from_events(table, args.events, float(args.min_sound_overlap_s))

    table["score_sound"] = scores
    table["score_sound_std"] = score_std
    table["pred"] = pred
    table["pred_label"] = np.where(pred == 1, "sound", "noise")
    table.to_csv(args.output_dir / "window_predictions.csv", index=False)

    events = _merge_events(table, threshold, float(args.merge_gap_s))
    events.to_csv(args.output_dir / "predicted_sound_events.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "ensemble_heads.csv", index=False)

    metrics: dict[str, float | int | str] = {
        "model_name": args.model_name,
        "heads_dir": str(args.heads_dir),
        "n_heads": int(fold_scores.shape[0]),
        "threshold": float(threshold),
        "merge_gap_s": float(args.merge_gap_s),
        "windows": int(len(table)),
        "predicted_sound_windows": int(pred.sum()),
        "predicted_sound_events": int(len(events)),
    }
    if "label" in table.columns and table["label"].notna().any() and table["label"].nunique() > 1:
        y_true = table["label"].astype(int).to_numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
        metrics.update(
            {
                "window_precision_sound": float(precision[1]),
                "window_recall_sound": float(recall[1]),
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
    parser.add_argument("--merge-gap-s", type=float, default=0.0)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--min-sound-overlap-s", type=float, default=0.25)
    parser.add_argument("--segment-duration-s", type=float, default=60.0)
    return parser.parse_args()


if __name__ == "__main__":
    apply_heads(parse_args())
