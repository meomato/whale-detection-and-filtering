"""Train detector-style heads on frozen encoder embeddings.

This is the cheap fallback for model comparison: the encoder is already frozen
and exported to ``embeddings.npy``. We only train a small binary head, select the
best checkpoint by validation event F1@0.5, then select the final detector
threshold on the same validation event F1@0.5 sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from filtering.finetune.detector_metrics import detector_metrics, select_threshold, write_predicted_events


def _load_sound_events(path: Path) -> dict[str, list[tuple[float, float]]]:
    ann = pd.read_csv(path)
    out: dict[str, list[tuple[float, float]]] = {}
    for row in ann.itertuples(index=False):
        if str(getattr(row, "label")).strip().lower() != "sound":
            continue
        out.setdefault(str(getattr(row, "audio")), []).append((float(getattr(row, "start_s")), float(getattr(row, "end_s"))))
    for filename in out:
        out[filename].sort()
    return out


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
    embeddings = np.load(args.embeddings).astype(np.float32)
    manifest = pd.read_csv(args.embedding_manifest)
    if len(embeddings) != len(manifest):
        raise ValueError(f"embeddings rows ({len(embeddings)}) != manifest rows ({len(manifest)})")
    for column in ("filename", "start_s", "end_s"):
        if column not in manifest.columns:
            raise ValueError(f"embedding manifest missing required column: {column}")

    splits = pd.read_csv(args.splits)
    split_map = dict(zip(splits["filename"].astype(str), splits["split"].astype(str)))
    events = _load_sound_events(args.annotations)

    table = manifest.copy()
    table["split"] = table["filename"].astype(str).map(split_map).fillna("ignore")
    table["label"] = [
        "sound"
        if _has_sound_overlap(
            str(row.filename),
            float(row.start_s),
            float(row.end_s),
            events,
            float(args.min_sound_overlap_s),
        )
        else "noise"
        for row in table.itertuples(index=False)
    ]
    table["label_id"] = (table["label"] == "sound").astype(np.int64)
    duration = table["end_s"].astype(float) - table["start_s"].astype(float)
    keep = table["split"].isin(["train", "val", "test"]) & duration.ge(float(args.min_window_duration_s))
    return embeddings[keep.to_numpy()], table.loc[keep].reset_index(drop=True)


def _split_arrays(embeddings: np.ndarray, table: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    mask = table["split"].eq(split).to_numpy()
    split_table = table.loc[mask].reset_index(drop=True)
    return embeddings[mask], split_table["label_id"].to_numpy(dtype=np.float32), split_table


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class BinaryHead(torch.nn.Module):
    def __init__(self, input_dim: int, head_kind: str, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if head_kind == "linear":
            self.net = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(input_dim, 1))
        elif head_kind == "mlp2":
            mid_dim = max(16, hidden_dim // 2)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, mid_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(mid_dim, 1),
            )
        else:
            raise ValueError(f"Unsupported head kind: {head_kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = y_true.astype(np.int64)
    pred = (scores >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out: dict[str, float | int] = {
        "accuracy": float((tp + tn) / max(1, tp + fp + tn + fn)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    if len(np.unique(y)) == 2:
        out["ap"] = float(average_precision_score(y, scores))
        out["roc_auc"] = float(roc_auc_score(y, scores))
    else:
        out["ap"] = 0.0
        out["roc_auc"] = 0.0
    return out


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: torch.device) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        EmbeddingDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    ys: list[float] = []
    scores: list[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ys.extend(y.detach().cpu().numpy().tolist())
        scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return float(np.mean(losses)) if losses else 0.0, np.asarray(ys, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    embeddings, table = _build_table(args)
    x_train, y_train, train_table = _split_arrays(embeddings, table, "train")
    x_val, y_val, val_table = _split_arrays(embeddings, table, "val")
    x_test, y_test, test_table = _split_arrays(embeddings, table, "test")
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("train, val, and test splits must all be non-empty")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = BinaryHead(x_train.shape[1], args.head_kind, args.hidden_dim, args.dropout).to(device)
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight if args.class_weight == "balanced" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = _make_loader(x_train, y_train, args.batch_size, True, device)
    val_loader = _make_loader(x_val, y_val, args.batch_size, False, device)
    test_loader = _make_loader(x_test, y_test, args.batch_size, False, device)

    best_score = -1.0
    best_epoch = 0
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _y_train_epoch, _s_train_epoch = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_val_epoch, s_val_epoch = _run_epoch(model, val_loader, criterion, device)
        val_metrics = _metrics(y_val_epoch, s_val_epoch, args.training_threshold)
        val_detector = detector_metrics(
            val_table,
            s_val_epoch,
            args.annotations,
            args.training_threshold,
            args.event_merge_gap_s,
            (0.5, 0.8),
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "val_event_f1_0_5": float(val_detector["event_f1_0_5"]),
            "val_event_precision_0_5": float(val_detector["event_precision_0_5"]),
            "val_event_recall_0_5": float(val_detector["event_recall_0_5"]),
            "val_event_ap_0_5": float(val_detector["event_ap_0_5"]),
            "val_event_f1_0_8": float(val_detector["event_f1_0_8"]),
            "val_event_ap_0_8": float(val_detector["event_ap_0_8"]),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        score = float(row[f"val_{args.selection_metric}"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if epoch >= args.min_epochs and stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, y_val_final, s_val = _run_epoch(model, val_loader, criterion, device)
    selected_threshold, val_selected_detector, val_sweep = select_threshold(
        val_table,
        s_val,
        args.annotations,
        args.event_merge_gap_s,
        0.5,
        args.threshold_sweep_steps,
    )
    test_loss, y_test_final, s_test = _run_epoch(model, test_loader, criterion, device)
    val_metrics = _metrics(y_val_final, s_val, selected_threshold)
    test_metrics = _metrics(y_test_final, s_test, selected_threshold)
    test_detector = detector_metrics(
        test_table,
        s_test,
        args.annotations,
        selected_threshold,
        args.event_merge_gap_s,
        (0.5, 0.8),
    )

    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(val_sweep).to_csv(args.output_dir / "val_threshold_sweep_iou_0_5.csv", index=False)
    for key in ("sweep_0_5", "sweep_0_8"):
        pd.DataFrame(test_detector[key]).to_csv(args.output_dir / f"test_detector_{key}.csv", index=False)

    with (args.output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "start_s", "end_s", "label", "label_name", "split", "score_sound", "score", "prediction"],
        )
        writer.writeheader()
        for row, score in zip(test_table.itertuples(index=False), s_test, strict=False):
            writer.writerow(
                {
                    "filename": str(row.filename),
                    "start_s": float(row.start_s),
                    "end_s": float(row.end_s),
                    "label": int(row.label_id),
                    "label_name": str(row.label),
                    "split": str(row.split),
                    "score_sound": float(score),
                    "score": float(score),
                    "prediction": "sound" if score >= selected_threshold else "noise",
                }
            )
    write_predicted_events(
        args.output_dir / "test_predicted_events.csv",
        test_table,
        s_test,
        selected_threshold,
        args.event_merge_gap_s,
    )
    torch.save({"head": model.state_dict(), "args": vars(args), "epoch": best_epoch}, args.output_dir / "best_head.pt")
    joblib.dump({"scaler": scaler}, args.output_dir / "scaler.joblib")

    detection_summary = {
        "event_f1": float(test_detector.get("event_f1_0_5", 0.0)),
        "event_precision": float(test_detector.get("event_precision_0_5", 0.0)),
        "event_recall": float(test_detector.get("event_recall_0_5", 0.0)),
        "event_false_alarm_rate_per_hour": float(test_detector.get("event_false_alarm_rate_per_hour_0_5", 0.0)),
        "voxaboxen_style_event_ap_0.5": float(test_detector.get("event_ap_0_5", 0.0)),
        "voxaboxen_style_event_ap_0.8": float(test_detector.get("event_ap_0_8", 0.0)),
    }
    summary = {
        "model": args.model_name,
        "model_name": args.model_name,
        "scenario": args.scenario,
        "head_kind": args.head_kind,
        "embedding_dim": int(x_train.shape[1]),
        "device": str(device),
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(len(rows)),
        "checkpoint_selected_on": f"val_{args.selection_metric}",
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "selected_threshold": float(selected_threshold),
        "threshold_selected_on": "val_event_f1_0_5",
        "train_windows": int(len(train_table)),
        "val_windows": int(len(val_table)),
        "test_windows": int(len(test_table)),
        "val_loss": float(val_loss),
        "val": val_metrics,
        "val_selected_detector": val_selected_detector,
        "test_loss": float(test_loss),
        "test": test_metrics,
        "detection": detection_summary,
        "test_detector": {key: value for key, value in test_detector.items() if not str(key).startswith("sweep_")},
        "seconds": float(time.time() - started),
        "elapsed_s": float(time.time() - started),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "selected_threshold.json").write_text(
        json.dumps(
            {
                "selected_threshold": float(selected_threshold),
                "selected_on": "val_event_f1_0_5",
                "val_selected_detector": val_selected_detector,
                "threshold_sweep_steps": int(args.threshold_sweep_steps),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("summary", json.dumps(summary), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--scenario", default="annotations_all")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-kind", choices=["linear", "mlp2"], default="mlp2")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--training-threshold", type=float, default=0.5)
    parser.add_argument("--selection-metric", choices=["f1", "ap", "roc_auc", "event_f1_0_5", "event_ap_0_5"], default="event_f1_0_5")
    parser.add_argument("--event-merge-gap-s", type=float, default=0.0)
    parser.add_argument("--threshold-sweep-steps", type=int, default=101)
    parser.add_argument("--min-sound-overlap-s", type=float, default=0.1)
    parser.add_argument("--min-window-duration-s", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
