"""Train the same simple sound/noise head on saved embeddings.

Input files:
- ``embeddings.npy``: shape [N, D]
- ``embedding_manifest.csv``: at least filename,start_s,end_s

This keeps the encoders frozen and only trains a small logistic head.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


CLASSES = np.array([0, 1], dtype=np.int64)
CLASS_NAMES = ["noise", "sound"]
COLOR_F1 = "#0F4C5C"
COLOR_PRECISION = "#D97757"
COLOR_RECALL = "#2A9D8F"
COLOR_CURVE = "#0F4C5C"
COLOR_PR = "#D97757"
COLOR_GRID = "#DCE6E8"
COLOR_CONFUSION = "YlGnBu"
MODEL_LABELS = {
    "animal2vec_pretrained_meerkat": "animal2vec",
    "perch_v2": "Perch 2.0",
    "voxaboxen_beats": "Voxaboxen",
    "wav2vec2_base": "Wav2Vec2",
}
SCENARIO_LABELS = {
    "annotations_all": "all labels",
    "annotations_v1": "labels v1",
    "annotations_v2": "labels v2",
}


def _pretty_title(args: argparse.Namespace) -> str:
    model = MODEL_LABELS.get(args.model_name, args.model_name)
    scenario = SCENARIO_LABELS.get(args.scenario, args.scenario)
    return f"{model} - {scenario}"


def _load_sound_events(path: Path) -> dict[str, list[tuple[float, float]]]:
    ann = pd.read_csv(path)
    events: dict[str, list[tuple[float, float]]] = {}
    for _, row in ann.iterrows():
        if str(row["label"]).strip().lower() != "sound":
            continue
        events.setdefault(str(row["audio"]), []).append((float(row["start_s"]), float(row["end_s"])))
    for filename in events:
        events[filename].sort()
    return events


def _merge_intervals(intervals: list[tuple[float, float]], max_gap_s: float = 0.0) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted((float(start), float(end)) for start, end in intervals if float(end) > float(start))
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_gap_s:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _events_from_windows(
    windows: pd.DataFrame,
    positive_mask: np.ndarray,
    max_gap_s: float,
) -> dict[str, list[tuple[float, float]]]:
    events: dict[str, list[tuple[float, float]]] = {}
    if len(windows) == 0:
        return events
    tmp = windows.loc[positive_mask, ["filename", "start_s", "end_s"]].copy()
    for filename, group in tmp.groupby("filename", sort=False):
        intervals = [(float(row.start_s), float(row.end_s)) for row in group.itertuples(index=False)]
        events[str(filename)] = _merge_intervals(intervals, max_gap_s=max_gap_s)
    return events


def _clip_events_to_windows(
    events: dict[str, list[tuple[float, float]]],
    windows: pd.DataFrame,
) -> dict[str, list[tuple[float, float]]]:
    clipped: dict[str, list[tuple[float, float]]] = {}
    for filename, group in windows.groupby("filename", sort=False):
        file_start = float(group["start_s"].min())
        file_end = float(group["end_s"].max())
        intervals: list[tuple[float, float]] = []
        for start, end in events.get(str(filename), []):
            overlap_start = max(file_start, float(start))
            overlap_end = min(file_end, float(end))
            if overlap_end > overlap_start:
                intervals.append((overlap_start, overlap_end))
        clipped[str(filename)] = _merge_intervals(intervals)
    return clipped


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if overlap <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(overlap / union) if union > 0 else 0.0


def _match_events(
    true_events: dict[str, list[tuple[float, float]]],
    pred_events: dict[str, list[tuple[float, float]]],
    iou_threshold: float,
) -> tuple[int, int, int, list[float], list[float], list[float]]:
    tp = fp = fn = 0
    matched_ious: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    filenames = sorted(set(true_events) | set(pred_events))
    for filename in filenames:
        true_list = true_events.get(filename, [])
        pred_list = pred_events.get(filename, [])
        candidates: list[tuple[float, int, int]] = []
        for pred_idx, pred in enumerate(pred_list):
            for true_idx, true in enumerate(true_list):
                iou = _interval_iou(pred, true)
                if iou >= iou_threshold:
                    candidates.append((iou, pred_idx, true_idx))
        candidates.sort(reverse=True)
        used_pred: set[int] = set()
        used_true: set[int] = set()
        for iou, pred_idx, true_idx in candidates:
            if pred_idx in used_pred or true_idx in used_true:
                continue
            used_pred.add(pred_idx)
            used_true.add(true_idx)
            matched_ious.append(float(iou))
            pred = pred_list[pred_idx]
            true = true_list[true_idx]
            start_errors.append(abs(float(pred[0]) - float(true[0])))
            end_errors.append(abs(float(pred[1]) - float(true[1])))
        tp += len(used_true)
        fp += len(pred_list) - len(used_pred)
        fn += len(true_list) - len(used_true)
    return tp, fp, fn, matched_ious, start_errors, end_errors


def _event_metrics_for_threshold(
    test_table: pd.DataFrame,
    scores: np.ndarray,
    true_events: dict[str, list[tuple[float, float]]],
    threshold: float,
    iou_threshold: float,
    max_gap_s: float,
) -> dict:
    pred_events = _events_from_windows(test_table, scores >= threshold, max_gap_s=max_gap_s)
    tp, fp, fn, ious, start_errors, end_errors = _match_events(true_events, pred_events, iou_threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    if "duration_s" in test_table.columns:
        evaluated_seconds = float(test_table.drop_duplicates("filename")["duration_s"].astype(float).sum())
    else:
        evaluated_seconds = 0.0
        for _, group in test_table.groupby("filename", sort=False):
            evaluated_seconds += float(group["end_s"].astype(float).max() - group["start_s"].astype(float).min())
    evaluated_hours = evaluated_seconds / 3600.0
    pred_sound_seconds = sum(end - start for events in pred_events.values() for start, end in events)
    return {
        "threshold": float(threshold),
        "event_iou_threshold": float(iou_threshold),
        "event_tp": int(tp),
        "event_fp": int(fp),
        "event_fn": int(fn),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_false_alarm_rate_per_hour": float(fp / evaluated_hours) if evaluated_hours > 0 else 0.0,
        "event_false_alarm_rate_per_eval_hour": float(fp / evaluated_hours) if evaluated_hours > 0 else 0.0,
        "mean_event_iou": float(np.mean(ious)) if ious else 0.0,
        "median_start_error_s": float(np.median(start_errors)) if start_errors else None,
        "median_end_error_s": float(np.median(end_errors)) if end_errors else None,
        "predicted_sound_minutes_per_hour": float((pred_sound_seconds / 60.0) / evaluated_hours)
        if evaluated_hours > 0
        else 0.0,
        "evaluated_hours": evaluated_hours,
        "evaluated_noise_hours": evaluated_hours,
    }


def _recall_at_far(
    test_table: pd.DataFrame,
    scores: np.ndarray,
    true_events: dict[str, list[tuple[float, float]]],
    far_limits: tuple[float, ...],
    iou_threshold: float,
    max_gap_s: float,
) -> dict[str, float | None]:
    thresholds = np.linspace(0.0, 1.0, 101)
    out: dict[str, float | None] = {f"event_recall_at_far_{limit:g}_per_hour": None for limit in far_limits}
    best: dict[float, float] = {limit: -1.0 for limit in far_limits}
    for threshold in thresholds:
        metrics = _event_metrics_for_threshold(test_table, scores, true_events, float(threshold), iou_threshold, max_gap_s)
        far = metrics["event_false_alarm_rate_per_hour"]
        for limit in far_limits:
            if far <= limit and metrics["event_recall"] > best[limit]:
                best[limit] = float(metrics["event_recall"])
                out[f"event_recall_at_far_{limit:g}_per_hour"] = float(metrics["event_recall"])
    return out


def _interpolated_ap(precision: list[float], recall: list[float]) -> float:
    if not precision or not recall:
        return 0.0
    precs = [0.0] + [float(x) for x in precision] + [1.0]
    recs = [1.0] + [float(x) for x in recall] + [0.0]
    best_by_recall: dict[float, float] = {}
    for r, p in zip(recs, precs, strict=False):
        best_by_recall[r] = max(p, best_by_recall.get(r, 0.0))
    recs_np = np.asarray(list(best_by_recall.keys()), dtype=float)
    precs_np = np.asarray(list(best_by_recall.values()), dtype=float)
    total = 0.0
    recall_levels = np.linspace(0, 1, 1001)
    for recall_level in recall_levels:
        mask = recs_np >= recall_level
        total += float(precs_np[mask].max()) if mask.any() else 0.0
    return total / len(recall_levels)


def _event_average_precision(
    test_table: pd.DataFrame,
    scores: np.ndarray,
    true_events: dict[str, list[tuple[float, float]]],
    iou_threshold: float,
    max_gap_s: float,
    n_thresholds: int = 101,
) -> dict:
    thresholds = np.linspace(0.0, 1.0, int(n_thresholds))
    rows: list[dict] = []
    precision: list[float] = []
    recall: list[float] = []
    for threshold in thresholds:
        metrics = _event_metrics_for_threshold(
            test_table,
            scores,
            true_events,
            float(threshold),
            iou_threshold,
            max_gap_s,
        )
        if metrics["event_tp"] + metrics["event_fp"] + metrics["event_fn"] == 0:
            continue
        precision.append(float(metrics["event_precision"]))
        recall.append(float(metrics["event_recall"]))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(metrics["event_precision"]),
                "recall": float(metrics["event_recall"]),
                "tp": int(metrics["event_tp"]),
                "fp": int(metrics["event_fp"]),
                "fn": int(metrics["event_fn"]),
            }
        )
    return {
        "ap": float(_interpolated_ap(precision, recall)),
        "iou_threshold": float(iou_threshold),
        "n_thresholds": int(n_thresholds),
        "sweep": rows,
    }


def _detection_metrics(
    test_table: pd.DataFrame,
    scores: np.ndarray,
    annotations: Path,
    iou_threshold: float,
    max_gap_s: float,
    threshold: float,
) -> dict:
    all_true_events = _load_sound_events(annotations)
    true_events = _clip_events_to_windows(all_true_events, test_table)
    metrics = _event_metrics_for_threshold(test_table, scores, true_events, threshold, iou_threshold, max_gap_s)
    metrics.update(_recall_at_far(test_table, scores, true_events, (1.0, 5.0, 10.0), iou_threshold, max_gap_s))
    event_ap_05 = _event_average_precision(test_table, scores, true_events, 0.5, max_gap_s)
    event_ap_08 = _event_average_precision(test_table, scores, true_events, 0.8, max_gap_s)
    metrics["voxaboxen_style_event_ap_0.5"] = event_ap_05["ap"]
    metrics["voxaboxen_style_event_ap_0.8"] = event_ap_08["ap"]
    metrics["voxaboxen_style_event_ap"] = {
        "0.5": event_ap_05,
        "0.8": event_ap_08,
    }

    y_true = test_table["label"].to_numpy()
    y_pred = (scores >= threshold).astype(int)
    if "duration_s" in test_table.columns:
        evaluated_hours = float(test_table.drop_duplicates("filename")["duration_s"].astype(float).sum() / 3600.0)
    else:
        evaluated_hours = float(
            sum(
                group["end_s"].astype(float).max() - group["start_s"].astype(float).min()
                for _, group in test_table.groupby("filename", sort=False)
            )
            / 3600.0
        )
    false_positive_windows = int(((y_true == 0) & (y_pred == 1)).sum())
    metrics["window_false_alarm_rate_per_hour"] = float(false_positive_windows / evaluated_hours) if evaluated_hours > 0 else 0.0
    metrics["false_positive_windows"] = false_positive_windows
    return metrics


def _has_sound_overlap(
    filename: str,
    start_s: float,
    end_s: float,
    events: dict[str, list[tuple[float, float]]],
    min_overlap_s: float,
) -> bool:
    for event_start, event_end in events.get(filename, []):
        if event_start > end_s:
            break
        overlap = min(end_s, event_end) - max(start_s, event_start)
        if overlap >= min_overlap_s:
            return True
    return False


def _build_table(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(args.embeddings)
    manifest = pd.read_csv(args.embedding_manifest)
    if len(X) != len(manifest):
        raise ValueError(f"embeddings rows ({len(X)}) != manifest rows ({len(manifest)})")
    for column in ("filename", "start_s", "end_s"):
        if column not in manifest.columns:
            raise ValueError(f"embedding manifest missing required column: {column}")

    splits = pd.read_csv(args.splits)
    split_map = dict(zip(splits["filename"].astype(str), splits["split"].astype(str)))
    sound_events = _load_sound_events(args.annotations)

    labels: list[int] = []
    split_values: list[str] = []
    for _, row in manifest.iterrows():
        filename = str(row["filename"])
        labels.append(
            int(
                _has_sound_overlap(
                    filename,
                    float(row["start_s"]),
                    float(row["end_s"]),
                    sound_events,
                    float(args.min_sound_overlap_s),
                )
            )
        )
        split_values.append(split_map.get(filename, "ignore"))

    table = manifest.copy()
    table["label"] = labels
    table["label_name"] = np.where(table["label"].to_numpy() == 1, "sound", "noise")
    table["split"] = split_values
    duration = table["end_s"].astype(float) - table["start_s"].astype(float)
    keep = (
        table["split"].isin(["train", "val", "test"])
        & duration.ge(float(args.min_window_duration_s))
    ).to_numpy()
    return X[keep], table.loc[keep].reset_index(drop=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict:
    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_sound": float(p[1]),
        "recall_sound": float(r[1]),
        "f1_sound": float(f1[1]),
        "support_noise": int(support[0]),
        "support_sound": int(support[1]),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": cm.tolist(),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))
        out["map_sound"] = out["average_precision"]
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        for target_precision in (0.8, 0.9):
            mask = precision_curve >= target_precision
            out[f"recall_at_precision_{target_precision:.1f}"] = float(recall_curve[mask].max()) if mask.any() else 0.0
    return out


def _split_arrays(X: np.ndarray, table: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mask = table["split"].eq(split).to_numpy()
    return X[mask], table.loc[mask].reset_index(drop=True)["label"].to_numpy(), table.loc[mask].reset_index(drop=True)


def _score(model: SGDClassifier, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def _plot_epoch_history(history: list[dict], out_dir: Path, title: str) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#FBFCFC")
    ax.set_facecolor("#FBFCFC")
    lines = [
        ("val_f1_sound", "F1", COLOR_F1),
        ("val_precision_sound", "Precision", COLOR_PRECISION),
        ("val_recall_sound", "Recall", COLOR_RECALL),
    ]
    for metric, label, color in lines:
        ax.plot(epochs, [row.get(metric, 0.0) for row in history], marker="o", linewidth=2, label=label, color=color)
    ax.set_ylim(0, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation score")
    ax.set_title(f"{title}: validation by epoch", pad=10)
    ax.legend(frameon=False)
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "epoch_metrics.png", dpi=160)
    plt.close(fig)


def _plot_curves(y_true: np.ndarray, y_score: np.ndarray, out_dir: Path, title: str) -> None:
    if len(np.unique(y_true)) <= 1:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#FBFCFC")
    ax.set_facecolor("#FBFCFC")
    ax.plot(fpr, tpr, color=COLOR_CURVE, linewidth=2)
    ax.plot([0, 1], [0, 1], color="#A0A7B4", linestyle="--", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{title}: ROC", pad=10)
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#FBFCFC")
    ax.set_facecolor("#FBFCFC")
    ax.plot(recall, precision, color=COLOR_PR, linewidth=2)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"{title}: precision-recall", pad=10)
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(cm: list[list[int]], out_dir: Path, title: str) -> None:
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor="#FBFCFC")
    ax.set_facecolor("#FBFCFC")
    ax.imshow(arr, cmap=COLOR_CONFUSION)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}: confusion matrix", pad=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(arr[i, j])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    X, table = _build_table(args)
    X_train, y_train, train_table = _split_arrays(X, table, "train")
    X_val, y_val, _ = _split_arrays(X, table, "val")
    X_test, y_test, test_table = _split_arrays(X, table, "test")
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train and test splits must be non-empty.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val) if len(X_val) else X_val
    X_test = scaler.transform(X_test)

    present_classes = np.unique(y_train)
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train,
    )
    class_weights = {
        int(cls): float(weight) for cls, weight in zip(present_classes, class_weights_arr)
    }

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(args.alpha),
        learning_rate="constant",
        eta0=float(args.learning_rate),
        class_weight=class_weights,
        random_state=int(args.seed),
    )

    history: list[dict] = []
    best_metric = -1.0
    best_epoch = 0
    best_bundle = None
    stale_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.partial_fit(X_train, y_train, classes=CLASSES)
        val_X = X_val if len(X_val) else X_train
        val_y = y_val if len(X_val) else y_train
        train_pred = model.predict(X_train)
        val_pred = model.predict(val_X)
        val_score = _score(model, val_X)
        train_metrics = _metrics(y_train, train_pred, _score(model, X_train))
        val_metrics = _metrics(val_y, val_pred, val_score)
        row = {
            "epoch": epoch,
            "train_f1_sound": train_metrics["f1_sound"],
            "train_precision_sound": train_metrics["precision_sound"],
            "train_recall_sound": train_metrics["recall_sound"],
            "val_f1_sound": val_metrics["f1_sound"],
            "val_precision_sound": val_metrics["precision_sound"],
            "val_recall_sound": val_metrics["recall_sound"],
            "val_average_precision": val_metrics.get("average_precision"),
            "val_roc_auc": val_metrics.get("roc_auc"),
        }
        history.append(row)
        metric = float(val_metrics.get("average_precision") or val_metrics["f1_sound"])
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_bundle = {"model": copy.deepcopy(model), "scaler": copy.deepcopy(scaler)}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(args.patience):
            break

    if best_bundle is not None:
        bundle = best_bundle
        model = bundle["model"]
        scaler = bundle["scaler"]

    test_score = _score(model, X_test)
    test_pred = model.predict(X_test)
    test_metrics = _metrics(y_test, test_pred, test_score)
    detection_metrics = _detection_metrics(
        test_table,
        test_score,
        args.annotations,
        float(args.event_iou_threshold),
        float(args.event_merge_gap_s),
        float(args.decision_threshold),
    )

    predictions = test_table.copy()
    predictions["score_sound"] = test_score
    predictions["pred_label"] = np.where(test_pred == 1, "sound", "noise")
    predictions.to_csv(out_dir / "test_predictions.csv", index=False)
    train_table.to_csv(out_dir / "train_windows.csv", index=False)
    table.to_csv(out_dir / "all_labeled_windows.csv", index=False)

    with (out_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)

    elapsed_s = time.perf_counter() - started
    result = {
        "model_name": args.model_name,
        "scenario": args.scenario,
        "embedding_kind": args.embedding_kind,
        "embeddings": str(args.embeddings),
        "embedding_manifest": str(args.embedding_manifest),
        "annotations": str(args.annotations),
        "splits": str(args.splits),
        "classes": CLASS_NAMES,
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "learning_rate": float(args.learning_rate),
        "alpha": float(args.alpha),
        "class_weight": class_weights,
        "device": "cpu",
        "elapsed_s": elapsed_s,
        "split_counts": {
            "train": int((table["split"] == "train").sum()),
            "val": int((table["split"] == "val").sum()),
            "test": int((table["split"] == "test").sum()),
        },
        "test": test_metrics,
        "detection": detection_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "scaler": scaler, "class_names": CLASS_NAMES}, out_dir / "model.joblib")
    title = _pretty_title(args)
    _plot_epoch_history(history, out_dir, title)
    _plot_curves(y_test, test_score, out_dir, title)
    _plot_confusion_matrix(test_metrics["confusion_matrix"], out_dir, title)
    print(json.dumps({"output_dir": str(out_dir), "test": test_metrics, "detection": detection_metrics}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--embedding-kind", default="frozen_encoder")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-sound-overlap-s", type=float, default=0.1)
    parser.add_argument("--min-window-duration-s", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--event-iou-threshold", type=float, default=0.1)
    parser.add_argument("--event-merge-gap-s", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
