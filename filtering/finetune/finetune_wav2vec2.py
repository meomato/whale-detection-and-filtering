"""Fine-tune Wav2Vec2 on whale sound/noise windows."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import Wav2Vec2Model

from filtering.benchmark.audio_paths import resolve_audio_path
from filtering.finetune.detector_metrics import detector_metrics, select_threshold, write_predicted_events


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
    mono = torch.from_numpy(data.mean(axis=1)).float() if data.size else torch.zeros(0, dtype=torch.float32)
    if sr != target_sr and mono.numel():
        mono = AF.resample(mono, orig_freq=int(sr), new_freq=int(target_sr))
    target_len = int(round(duration_s * target_sr))
    if mono.numel() < target_len:
        mono = torch.nn.functional.pad(mono, (0, target_len - mono.numel()))
    elif mono.numel() > target_len:
        mono = mono[:target_len]
    return mono.float()


def _normalize(wav: torch.Tensor) -> torch.Tensor:
    mean = wav.mean(dim=1, keepdim=True)
    std = wav.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (wav - mean) / std


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, table: pd.DataFrame, audio_dir: Path, sample_rate: int, window_size_s: float) -> None:
        self.table = table.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sample_rate = int(sample_rate)
        self.window_size_s = float(window_size_s)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
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


class Wav2Vec2BinaryClassifier(torch.nn.Module):
    def __init__(self, model_id: str, dropout: float) -> None:
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_id)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_values=input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(self.dropout(pooled)).squeeze(1)


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = y_true.astype(int)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out: dict[str, float | int] = {
        "accuracy": (tp + tn) / max(1, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
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


def _load_table(path: Path, max_rows: int | None) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table[table["label"].isin(["sound", "noise"])].copy()
    table["label_id"] = (table["label"] == "sound").astype(int)
    if max_rows and len(table) > max_rows:
        table = table.sample(max_rows, random_state=13).sort_values(["filename", "start_s"])
    return table.reset_index(drop=True)


def _make_loader(table: pd.DataFrame, args: argparse.Namespace, shuffle: bool, device: torch.device):
    return torch.utils.data.DataLoader(
        WindowDataset(table, args.audio_dir, args.sample_rate, args.window_size_s),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )


def _run_epoch(model, loader, optimizer, device, args, train: bool):
    model.train(train)
    criterion = torch.nn.BCEWithLogitsLoss()
    losses: list[float] = []
    ys: list[float] = []
    scores: list[float] = []
    metas: list[dict[str, object]] = []
    for audio, labels, meta in loader:
        audio = audio.to(device)
        labels = labels.to(device)
        if args.normalize:
            audio = _normalize(audio)
        with torch.set_grad_enabled(train):
            logits = model(audio)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ys.extend(labels.detach().cpu().numpy().tolist())
        scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        metas.extend(meta)
    return float(np.mean(losses)) if losses else 0.0, np.asarray(ys), np.asarray(scores), metas


def train(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_table = _load_table(args.train_windows, args.max_train_windows)
    val_table = _load_table(args.val_windows, args.max_val_windows)
    test_table = _load_table(args.test_windows, args.max_test_windows)

    train_loader = _make_loader(train_table, args, shuffle=True, device=device)
    val_loader = _make_loader(val_table, args, shuffle=False, device=device)
    test_loader = _make_loader(test_table, args, shuffle=False, device=device)

    model = Wav2Vec2BinaryClassifier(args.model_id, args.dropout).to(device)
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    if args.freeze_feature_encoder:
        model.encoder.freeze_feature_encoder()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_epoch = 0
    stale = 0
    rows: list[dict[str, float | int]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, _y_train, _s_train, _ = _run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_loss, y_val, s_val, _ = _run_epoch(model, val_loader, optimizer, device, args, train=False)
        val_metrics = _metrics(y_val, s_val, args.threshold)
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
            "val_loss": val_loss,
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                args.output_dir / "best_checkpoint.pt",
            )
        else:
            stale += 1
            if epoch >= args.min_epochs and stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    checkpoint = torch.load(args.output_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    val_loss, y_val, s_val, _ = _run_epoch(model, val_loader, optimizer, device, args, train=False)
    selected_threshold, val_selected_metrics, val_threshold_sweep = select_threshold(
        val_table,
        s_val,
        args.annotations,
        args.event_merge_gap_s,
        0.5,
        args.threshold_sweep_steps,
    )
    test_loss, y_test, s_test, metas = _run_epoch(model, test_loader, optimizer, device, args, train=False)
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
        "model": "wav2vec2_full",
        "model_id": args.model_id,
        "device": str(device),
        "train_windows": int(len(train_table)),
        "val_windows": int(len(val_table)),
        "test_windows": int(len(test_table)),
        "epochs_requested": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "checkpoint_selected_on": f"val_{args.selection_metric}",
        "selection_metric": args.selection_metric,
        "training_threshold": float(args.threshold),
        "selected_threshold": float(selected_threshold),
        "threshold_selected_on": "val_event_f1_0_5",
        "val_loss": float(val_loss),
        "val": val_metrics,
        "val_selected_detector": val_selected_metrics,
        "seconds": time.time() - started,
        "test_loss": float(test_loss),
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
    parser.add_argument("--model-id", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--audio-dir", type=Path, default=Path("data/benchmark_audio"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-windows", type=Path, required=True)
    parser.add_argument("--val-windows", type=Path, required=True)
    parser.add_argument("--test-windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-size-s", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--event-merge-gap-s", type=float, default=0.0)
    parser.add_argument("--threshold-sweep-steps", type=int, default=101)
    parser.add_argument(
        "--selection-metric",
        choices=["ap", "f1", "roc_auc", "event_f1_0_5", "event_ap_0_5"],
        default="event_f1_0_5",
    )
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-val-windows", type=int)
    parser.add_argument("--max-test-windows", type=int)
    parser.add_argument("--freeze-feature-encoder", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
