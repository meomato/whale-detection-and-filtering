"""Fine-tune animal2vec on whale sound/noise windows."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.machinery
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

from filtering.benchmark.audio_paths import resolve_audio_path
from filtering.finetune.detector_metrics import detector_metrics, select_threshold, write_predicted_events


def _clone_state_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value(item) for item in value)
    return copy.deepcopy(value)


def _state_to_cpu(module: torch.nn.Module) -> dict[str, object]:
    return {key: _clone_state_value(value) for key, value in module.state_dict().items()}


def _load_model(animal2vec_dir: Path, checkpoint: Path, device: torch.device):
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    if "tensorflow" not in sys.modules:
        tf_stub = types.ModuleType("tensorflow")
        tf_image_stub = types.ModuleType("tensorflow.image")
        tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
        tf_image_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow.image", loader=None)

        def _decode_png_stub(*_args, **_kwargs):
            raise RuntimeError("tensorflow.image.decode_png is not available")

        tf_image_stub.decode_png = _decode_png_stub
        tf_stub.image = tf_image_stub
        sys.modules["tensorflow"] = tf_stub
        sys.modules["tensorflow.image"] = tf_image_stub

    sys.path.insert(0, str(animal2vec_dir))
    import nn  # noqa: F401
    from fairseq import checkpoint_utils

    models, _ = checkpoint_utils.load_model_ensemble([str(checkpoint)])
    model = models[0].to(device)
    model.train()
    return model


def _read_resampled_window(
    audio_dir: Path,
    filename: str,
    start_s: float,
    target_sr: int,
    duration_s: float,
) -> torch.Tensor:
    path = resolve_audio_path(audio_dir, filename)
    info = sf.info(str(path))
    start_frame = max(0, int(round(start_s * info.samplerate)))
    stop_frame = min(info.frames, start_frame + int(round(duration_s * info.samplerate)))
    data, sr = sf.read(str(path), start=start_frame, stop=stop_frame, dtype="float32", always_2d=True)
    if data.size == 0:
        mono = torch.zeros(0, dtype=torch.float32)
    else:
        mono = torch.from_numpy(data.mean(axis=1)).float()
    if sr != target_sr and mono.numel():
        mono = AF.resample(mono, orig_freq=int(sr), new_freq=int(target_sr))
    target_len = int(round(duration_s * target_sr))
    if mono.numel() < target_len:
        mono = torch.nn.functional.pad(mono, (0, target_len - mono.numel()))
    elif mono.numel() > target_len:
        mono = mono[:target_len]
    return mono.float()


def _normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.nn.functional.layer_norm(x, x.shape) for x in batch])


def _pool_features(features: dict[str, object], top_k_layers: int) -> torch.Tensor:
    layer_results = features["layer_results"]
    selected = layer_results[-top_k_layers:] if top_k_layers > 0 else layer_results
    tensors = []
    for layer in selected:
        tensor = layer[0] if isinstance(layer, tuple) else layer
        if tensor.dim() != 3:
            raise ValueError(f"Expected [B, T, C], got {tuple(tensor.shape)}")
        tensors.append(tensor.float())
    stacked = torch.stack(tensors, dim=0).mean(dim=0)
    return stacked.mean(dim=1)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, table: pd.DataFrame, audio_dir: Path, sample_rate: int, window_size_s: float) -> None:
        self.table = table.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sample_rate = int(sample_rate)
        self.window_size_s = float(window_size_s)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        row = self.table.iloc[index]
        audio = _read_resampled_window(
            self.audio_dir,
            str(row.filename),
            float(row.start_s),
            self.sample_rate,
            self.window_size_s,
        )
        label = torch.tensor(1.0 if row.label == "sound" else 0.0)
        meta = {
            "filename": str(row.filename),
            "start_s": float(row.start_s),
            "end_s": float(row.end_s),
            "label": str(row.label),
            "split": str(row.split),
        }
        return audio, label, meta


def _collate(batch):
    audio, labels, meta = zip(*batch)
    return torch.stack(audio), torch.stack(labels).float(), list(meta)


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    y = y_true.astype(int)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _load_table(path: Path, max_rows: int | None) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table[table["label"].isin(["sound", "noise"])].copy()
    table["label_id"] = (table["label"] == "sound").astype(int)
    if max_rows and len(table) > max_rows:
        table = table.sample(max_rows, random_state=13).sort_values(["filename", "start_s"])
    return table.reset_index(drop=True)


def _run_epoch(model, head, loader, optimizer, device, args) -> float:
    model.train()
    head.train()
    losses = []
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for audio, labels, _meta in loader:
        audio = audio.to(device)
        labels = labels.to(device)
        if args.normalize:
            audio = _normalize_batch(audio)
        optimizer.zero_grad(set_to_none=True)
        features = model.extract_features(source=audio)
        pooled = _pool_features(features, args.average_top_k_layers)
        logits = head(pooled).squeeze(1)
        loss = loss_fn(logits, labels)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def _predict(model, head, loader, device, args):
    model.eval()
    head.eval()
    ys: list[float] = []
    scores: list[float] = []
    metas: list[dict[str, object]] = []
    for audio, labels, meta in loader:
        audio = audio.to(device)
        if args.normalize:
            audio = _normalize_batch(audio)
        features = model.extract_features(source=audio)
        pooled = _pool_features(features, args.average_top_k_layers)
        logits = head(pooled).squeeze(1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        scores.extend(prob.tolist())
        ys.extend(labels.numpy().tolist())
        metas.extend(meta)
    return np.asarray(ys, dtype=np.float32), np.asarray(scores, dtype=np.float32), metas


def train(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = _load_model(args.animal2vec_dir, args.checkpoint, device)

    train_table = _load_table(args.train_windows, args.max_train_windows)
    val_table = _load_table(args.val_windows, args.max_val_windows)
    test_table = _load_table(args.test_windows, args.max_test_windows)
    train_loader = torch.utils.data.DataLoader(
        WindowDataset(train_table, args.audio_dir, args.sample_rate, args.window_size_s),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        WindowDataset(val_table, args.audio_dir, args.sample_rate, args.window_size_s),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    test_loader = torch.utils.data.DataLoader(
        WindowDataset(test_table, args.audio_dir, args.sample_rate, args.window_size_s),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    with torch.no_grad():
        sample = next(iter(train_loader))[0][:1].to(device)
        if args.normalize:
            sample = _normalize_batch(sample)
        feature_dim = int(_pool_features(model.extract_features(source=sample), args.average_top_k_layers).shape[1])
    head = torch.nn.Linear(feature_dim, 1).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.encoder_lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    best_score = -1.0
    best_epoch = 0
    best_model_state: dict[str, object] | None = None
    best_head_state: dict[str, object] | None = None
    best_val_metrics: dict[str, float | int] | None = None
    stale = 0
    rows = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, head, train_loader, optimizer, device, args)
        y_val, s_val, _ = _predict(model, head, val_loader, device, args)
        val_metrics = _metrics(y_val, s_val)
        val_detector = detector_metrics(
            val_table,
            s_val,
            args.annotations,
            args.threshold,
            args.event_merge_gap_s,
            (0.5, 0.8),
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
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
            best_model_state = _state_to_cpu(model)
            best_head_state = _state_to_cpu(head)
            best_val_metrics = copy.deepcopy(val_metrics)
        else:
            stale += 1
            if epoch >= args.min_epochs and stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_model_state is not None and best_head_state is not None:
        model.load_state_dict(best_model_state)
        head.load_state_dict(best_head_state)
    y_val, s_val, _ = _predict(model, head, val_loader, device, args)
    selected_threshold, val_selected_metrics, val_threshold_sweep = select_threshold(
        val_table,
        s_val,
        args.annotations,
        args.event_merge_gap_s,
        0.5,
        args.threshold_sweep_steps,
    )
    y_test, s_test, metas = _predict(model, head, test_loader, device, args)
    val_metrics = _metrics(y_val, s_val, selected_threshold)
    test_metrics = _metrics(y_test, s_test, selected_threshold)
    test_detector = detector_metrics(
        test_table,
        s_test,
        args.annotations,
        selected_threshold,
        args.event_merge_gap_s,
        (0.5, 0.8),
    )
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    if best_head_state is not None:
        torch.save(
            {
                "head": best_head_state,
                "args": vars(args),
                "epoch": best_epoch,
                "feature_dim": feature_dim,
                "val_metrics": best_val_metrics,
                "note": "Full animal2vec encoder checkpoint is kept in memory during the job to avoid large disk writes.",
            },
            args.output_dir / "best_head_checkpoint.pt",
        )
    with (args.output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "start_s", "end_s", "label", "split", "score", "prediction"])
        writer.writeheader()
        for meta, score in zip(metas, s_test):
            writer.writerow({**meta, "score": float(score), "prediction": "sound" if score >= selected_threshold else "noise"})
    write_predicted_events(
        args.output_dir / "test_predicted_events.csv",
        test_table,
        s_test,
        selected_threshold,
        args.event_merge_gap_s,
    )
    pd.DataFrame(val_threshold_sweep).to_csv(args.output_dir / "val_threshold_sweep_iou_0_5.csv", index=False)
    summary = {
        "model": "animal2vec_large_pretrained_meerkat_full",
        "device": str(device),
        "train_windows": int(len(train_table)),
        "val_windows": int(len(val_table)),
        "test_windows": int(len(test_table)),
        "feature_dim": int(feature_dim),
        "epochs_requested": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "checkpoint_selected_on": f"val_{args.selection_metric}",
        "selection_metric": args.selection_metric,
        "training_threshold": float(args.threshold),
        "selected_threshold": float(selected_threshold),
        "threshold_selected_on": "val_event_f1_0_5",
        "val": val_metrics,
        "val_selected_detector": val_selected_metrics,
        "seconds": time.time() - started,
        "test": test_metrics,
        "test_detector": {
            k: v for k, v in test_detector.items() if not str(k).startswith("sweep_")
        },
    }
    for key in ("sweep_0_5", "sweep_0_8"):
        pd.DataFrame(test_detector[key]).to_csv(args.output_dir / f"test_detector_{key}.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "selected_threshold.json").write_text(
        json.dumps(
            {
                "selected_threshold": float(selected_threshold),
                "selected_on": "val_event_f1_0_5",
                "val_selected_detector": val_selected_metrics,
                "threshold_sweep_steps": int(args.threshold_sweep_steps),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("test", json.dumps(test_metrics), flush=True)
    print("test_detector", json.dumps(summary["test_detector"]), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal2vec-dir", type=Path, default=Path("~/whales/external/animal2vec").expanduser())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, default=Path("data/benchmark_audio"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-windows", type=Path, required=True)
    parser.add_argument("--val-windows", type=Path, required=True)
    parser.add_argument("--test-windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--window-size-s", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--event-merge-gap-s", type=float, default=0.0)
    parser.add_argument("--threshold-sweep-steps", type=int, default=101)
    parser.add_argument(
        "--selection-metric",
        choices=["f1", "event_f1_0_5", "event_ap_0_5"],
        default="event_f1_0_5",
    )
    parser.add_argument("--average-top-k-layers", type=int, default=12)
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-val-windows", type=int)
    parser.add_argument("--max-test-windows", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
