"""Apply a fine-tuned Wav2Vec2 detector checkpoint to long-file windows."""

from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from transformers import Wav2Vec2Config, Wav2Vec2Model

from filtering.finetune.finetune_wav2vec2 import _normalize
from filtering.review.apply_downstream_head import _merge_events


def _torch_load_hpc(path: Path, device: torch.device) -> dict:
    original_posix = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        return torch.load(path, map_location=device, weights_only=False)
    finally:
        pathlib.PosixPath = original_posix


def _read_window(audio_dir: Path, filename: str, start_s: float, sample_rate: int, duration_s: float) -> torch.Tensor:
    path = audio_dir / filename
    info = sf.info(str(path))
    start_frame = max(0, int(round(start_s * info.samplerate)))
    stop_frame = min(info.frames, start_frame + int(round(duration_s * info.samplerate)))
    data, sr = sf.read(str(path), start=start_frame, stop=stop_frame, dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1)).float() if data.size else torch.zeros(0, dtype=torch.float32)
    if sr != sample_rate and mono.numel():
        mono = AF.resample(mono, orig_freq=int(sr), new_freq=int(sample_rate))
    target_len = int(round(duration_s * sample_rate))
    if mono.numel() < target_len:
        mono = torch.nn.functional.pad(mono, (0, target_len - mono.numel()))
    elif mono.numel() > target_len:
        mono = mono[:target_len]
    return mono.float()


class LocalWav2Vec2BinaryClassifier(torch.nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.encoder = Wav2Vec2Model(Wav2Vec2Config())
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_values=input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(self.dropout(pooled)).squeeze(1)


class LongWindowDataset(torch.utils.data.Dataset):
    def __init__(self, table: pd.DataFrame, audio_dir: Path, sample_rate: int, window_size_s: float) -> None:
        self.table = table.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sample_rate = int(sample_rate)
        self.window_size_s = float(window_size_s)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        audio = _read_window(self.audio_dir, str(row.filename), float(row.start_s), self.sample_rate, self.window_size_s)
        return audio, int(row.label), index


def _collate(batch):
    audio, labels, indexes = zip(*batch)
    return torch.stack(audio), torch.tensor(labels, dtype=torch.long), list(indexes)


def apply_detector(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = _torch_load_hpc(args.checkpoint, device)
    ckpt_args = checkpoint.get("args", {})
    sample_rate = int(ckpt_args.get("sample_rate", args.sample_rate))
    window_size_s = float(ckpt_args.get("window_size_s", args.window_size_s))
    dropout = float(ckpt_args.get("dropout", args.dropout))
    normalize = bool(ckpt_args.get("normalize", True))
    threshold = float(args.threshold if args.threshold is not None else args.selected_threshold)

    model = LocalWav2Vec2BinaryClassifier(dropout).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    table = pd.read_csv(args.windows)
    dataset = LongWindowDataset(table, args.audio_dir, sample_rate, window_size_s)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    scores = np.zeros(len(table), dtype=np.float32)
    with torch.no_grad():
        for audio, _labels, indexes in loader:
            audio = audio.to(device)
            if normalize:
                audio = _normalize(audio)
            logits = model(audio)
            scores[np.asarray(indexes, dtype=int)] = torch.sigmoid(logits).detach().cpu().numpy()

    pred = (scores >= threshold).astype(int)
    table["score_sound"] = scores
    table["pred"] = pred
    table["pred_label"] = np.where(pred == 1, "sound", "noise")
    table.to_csv(args.output_dir / "window_predictions.csv", index=False)

    events = _merge_events(table, threshold, float(args.merge_gap_s))
    events.to_csv(args.output_dir / "predicted_sound_events.csv", index=False)

    metrics: dict[str, float | int | str] = {
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "threshold": threshold,
        "merge_gap_s": float(args.merge_gap_s),
        "windows": int(len(table)),
        "predicted_sound_windows": int(pred.sum()),
        "predicted_sound_events": int(len(events)),
    }
    if "label" in table.columns and table["label"].nunique() > 1:
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
    parser.add_argument("--model-name", default="wav2vec2_full")
    parser.add_argument("--model-id", default="facebook/wav2vec2-base")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-threshold", type=float, default=0.44)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--merge-gap-s", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-size-s", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    apply_detector(parse_args())
