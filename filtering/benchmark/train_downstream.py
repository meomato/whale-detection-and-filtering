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
COLOR_F1 = "#3A6EA5"
COLOR_PRECISION = "#E07A5F"
COLOR_RECALL = "#59A14F"
COLOR_CURVE = "#3A6EA5"
COLOR_PR = "#E07A5F"
COLOR_GRID = "#D8DEE9"
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
    keep = table["split"].isin(["train", "val", "test"]).to_numpy()
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
    fig, ax = plt.subplots(figsize=(7, 4))
    lines = [
        ("val_f1_sound", "F1", COLOR_F1),
        ("val_precision_sound", "Precision", COLOR_PRECISION),
        ("val_recall_sound", "Recall", COLOR_RECALL),
    ]
    for metric, label, color in lines:
        ax.plot(epochs, [row.get(metric, 0.0) for row in history], marker="o", label=label, color=color)
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

    fig, ax = plt.subplots(figsize=(5, 4))
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

    fig, ax = plt.subplots(figsize=(5, 4))
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
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(arr, cmap="YlGnBu")
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
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "scaler": scaler, "class_names": CLASS_NAMES}, out_dir / "model.joblib")
    title = _pretty_title(args)
    _plot_epoch_history(history, out_dir, title)
    _plot_curves(y_test, test_score, out_dir, title)
    _plot_confusion_matrix(test_metrics["confusion_matrix"], out_dir, title)
    print(json.dumps({"output_dir": str(out_dir), "test": test_metrics}, indent=2))


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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
