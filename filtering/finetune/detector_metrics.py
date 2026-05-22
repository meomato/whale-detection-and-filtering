"""Detector-style metrics for sliding-window fine-tuning runs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def load_sound_events(path: Path) -> dict[str, list[tuple[float, float]]]:
    ann = pd.read_csv(path)
    events: dict[str, list[tuple[float, float]]] = {}
    for row in ann.itertuples(index=False):
        label = str(getattr(row, "label")).strip().lower()
        if label != "sound":
            continue
        filename = str(getattr(row, "audio"))
        events.setdefault(filename, []).append((float(getattr(row, "start_s")), float(getattr(row, "end_s"))))
    for filename in events:
        events[filename] = merge_intervals(events[filename])
    return events


def merge_intervals(intervals: list[tuple[float, float]], max_gap_s: float = 0.0) -> list[tuple[float, float]]:
    valid = sorted((float(start), float(end)) for start, end in intervals if float(end) > float(start))
    if not valid:
        return []
    merged = [valid[0]]
    for start, end in valid[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_gap_s:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def events_from_windows(
    windows: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    merge_gap_s: float,
) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    if len(windows) == 0:
        return out
    tmp = windows.loc[np.asarray(scores) >= float(threshold), ["filename", "start_s", "end_s"]].copy()
    for filename, group in tmp.groupby("filename", sort=False):
        intervals = [(float(row.start_s), float(row.end_s)) for row in group.itertuples(index=False)]
        out[str(filename)] = merge_intervals(intervals, max_gap_s=merge_gap_s)
    return out


def clip_events_to_windows(
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
        clipped[str(filename)] = merge_intervals(intervals)
    return clipped


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if overlap <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(overlap / union) if union > 0 else 0.0


def match_events(
    true_events: dict[str, list[tuple[float, float]]],
    pred_events: dict[str, list[tuple[float, float]]],
    iou_threshold: float,
) -> tuple[int, int, int, list[float]]:
    tp = fp = fn = 0
    matched_ious: list[float] = []
    for filename in sorted(set(true_events) | set(pred_events)):
        true_list = true_events.get(filename, [])
        pred_list = pred_events.get(filename, [])
        candidates: list[tuple[float, int, int]] = []
        for pred_idx, pred in enumerate(pred_list):
            for true_idx, true in enumerate(true_list):
                iou = interval_iou(pred, true)
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
        tp += len(used_true)
        fp += len(pred_list) - len(used_pred)
        fn += len(true_list) - len(used_true)
    return tp, fp, fn, matched_ious


def event_metrics_for_threshold(
    windows: pd.DataFrame,
    scores: np.ndarray,
    true_events: dict[str, list[tuple[float, float]]],
    threshold: float,
    iou_threshold: float,
    merge_gap_s: float,
) -> dict[str, float | int]:
    pred_events = events_from_windows(windows, scores, threshold, merge_gap_s)
    tp, fp, fn, ious = match_events(true_events, pred_events, iou_threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    evaluated_seconds = 0.0
    for _, group in windows.groupby("filename", sort=False):
        evaluated_seconds += float(group["end_s"].astype(float).max() - group["start_s"].astype(float).min())
    evaluated_hours = evaluated_seconds / 3600.0
    return {
        "threshold": float(threshold),
        "event_iou": float(iou_threshold),
        "event_tp": int(tp),
        "event_fp": int(fp),
        "event_fn": int(fn),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_false_alarm_rate_per_hour": float(fp / evaluated_hours) if evaluated_hours > 0 else 0.0,
        "evaluated_hours": float(evaluated_hours),
        "mean_event_iou": float(np.mean(ious)) if ious else 0.0,
    }


def interpolated_ap(precision: list[float], recall: list[float]) -> float:
    if not precision or not recall:
        return 0.0
    recall_levels = np.linspace(0.0, 1.0, 1001)
    prec = np.asarray(precision, dtype=float)
    rec = np.asarray(recall, dtype=float)
    total = 0.0
    for level in recall_levels:
        mask = rec >= level
        total += float(prec[mask].max()) if mask.any() else 0.0
    return total / len(recall_levels)


def event_average_precision(
    windows: pd.DataFrame,
    scores: np.ndarray,
    true_events: dict[str, list[tuple[float, float]]],
    iou_threshold: float,
    merge_gap_s: float,
    n_thresholds: int = 101,
) -> tuple[float, list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    precision: list[float] = []
    recall: list[float] = []
    for threshold in np.linspace(0.0, 1.0, int(n_thresholds)):
        metrics = event_metrics_for_threshold(windows, scores, true_events, float(threshold), iou_threshold, merge_gap_s)
        precision.append(float(metrics["event_precision"]))
        recall.append(float(metrics["event_recall"]))
        rows.append(metrics)
    return float(interpolated_ap(precision, recall)), rows


def detector_metrics(
    windows: pd.DataFrame,
    scores: np.ndarray,
    annotations: Path,
    threshold: float,
    merge_gap_s: float,
    iou_thresholds: tuple[float, ...] = (0.5, 0.8),
) -> dict[str, object]:
    true_events = clip_events_to_windows(load_sound_events(annotations), windows)
    out: dict[str, object] = {}
    for iou in iou_thresholds:
        key = str(iou).replace(".", "_")
        metrics = event_metrics_for_threshold(windows, scores, true_events, threshold, iou, merge_gap_s)
        ap, sweep = event_average_precision(windows, scores, true_events, iou, merge_gap_s)
        for metric_key, value in metrics.items():
            out[f"{metric_key}_{key}"] = value
        out[f"event_ap_{key}"] = ap
        out[f"sweep_{key}"] = sweep
    return out


def select_threshold(
    windows: pd.DataFrame,
    scores: np.ndarray,
    annotations: Path,
    merge_gap_s: float,
    iou_threshold: float = 0.5,
    n_thresholds: int = 101,
) -> tuple[float, dict[str, float | int], list[dict[str, float | int]]]:
    true_events = clip_events_to_windows(load_sound_events(annotations), windows)
    sweep: list[dict[str, float | int]] = []
    for threshold in np.linspace(0.0, 1.0, int(n_thresholds)):
        sweep.append(
            event_metrics_for_threshold(
                windows,
                scores,
                true_events,
                float(threshold),
                float(iou_threshold),
                float(merge_gap_s),
            )
        )
    best = sorted(
        sweep,
        key=lambda row: (
            float(row["event_f1"]),
            float(row["event_recall"]),
            float(row["event_precision"]),
        ),
        reverse=True,
    )[0]
    return float(best["threshold"]), best, sweep


def write_predicted_events(
    path: Path,
    windows: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    merge_gap_s: float,
) -> None:
    events = events_from_windows(windows, scores, threshold, merge_gap_s)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "start_s", "end_s", "threshold"])
        writer.writeheader()
        for filename, intervals in events.items():
            for start_s, end_s in intervals:
                writer.writerow(
                    {
                        "filename": filename,
                        "start_s": float(start_s),
                        "end_s": float(end_s),
                        "threshold": float(threshold),
                    }
                )
