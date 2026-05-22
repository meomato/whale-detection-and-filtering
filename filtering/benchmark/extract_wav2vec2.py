"""Extract frozen Wav2Vec2 embeddings for the benchmark."""

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
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from filtering.benchmark.audio_paths import resolve_audio_path


def _read_resampled_file(audio_dir: Path, filename: str, target_sr: int) -> torch.Tensor:
    data, sr = sf.read(str(resolve_audio_path(audio_dir, filename)), dtype="float32", always_2d=True)
    if data.size == 0:
        return torch.zeros(0, dtype=torch.float32)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != target_sr:
        mono = AF.resample(mono, orig_freq=sr, new_freq=target_sr)
    return mono.float()


def _slice_window(
    audio: torch.Tensor,
    start_s: float,
    target_sr: int,
    fixed_duration_s: float,
) -> np.ndarray:
    expected = int(round(fixed_duration_s * target_sr))
    if audio.numel() == 0:
        return np.zeros(expected, dtype=np.float32)
    start_frame = max(0, int(round(start_s * target_sr)))
    stop_frame = start_frame + expected
    chunk = audio[start_frame:stop_frame]
    if chunk.numel() < expected:
        chunk = torch.nn.functional.pad(chunk, (0, expected - chunk.numel()))
    return chunk.numpy().astype(np.float32, copy=False)


def extract(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_id)
    model = Wav2Vec2Model.from_pretrained(args.model_id).to(device).eval()
    windows = pd.read_csv(args.windows)

    rows: list[dict[str, str | int | float]] = []
    vectors: list[np.ndarray] = []
    batch_audio: list[np.ndarray] = []
    batch_rows: list[dict[str, str | int | float]] = []

    def flush() -> None:
        if not batch_audio:
            return
        inputs = feature_extractor(
            batch_audio,
            sampling_rate=int(args.sample_rate),
            padding=True,
            return_attention_mask=False,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device)
        with torch.inference_mode():
            outputs = model(input_values=input_values)
            pooled = outputs.last_hidden_state.mean(dim=1)
            pooled_np = pooled.detach().cpu().numpy().astype(np.float32)
        vectors.extend([pooled_np[i] for i in range(pooled_np.shape[0])])
        rows.extend(batch_rows)
        batch_audio.clear()
        batch_rows.clear()

    processed = 0
    total = int(len(windows) if not args.max_windows else min(len(windows), args.max_windows))
    for filename, file_windows in windows.groupby("filename", sort=False):
        audio = _read_resampled_file(args.audio_dir, str(filename), int(args.sample_rate))
        for _, row in file_windows.iterrows():
            wav = _slice_window(
                audio,
                float(row["start_s"]),
                int(args.sample_rate),
                float(args.window_size_s),
            )
            batch_audio.append(wav)
            batch_rows.append(
                {
                    "row_index": len(rows) + len(batch_rows),
                    "filename": str(row["filename"]),
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                }
            )
            processed += 1
            if len(batch_audio) >= int(args.batch_size):
                flush()
            if processed % int(args.progress_every) == 0:
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "total": total,
                            "elapsed_s": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            if args.max_windows and processed >= int(args.max_windows):
                break
        del audio
        if args.max_windows and processed >= int(args.max_windows):
            break
    flush()

    X = np.stack(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)
    np.save(out_dir / "embeddings.npy", X)
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "filename", "start_s", "end_s"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "model_name": "wav2vec2_base",
        "model_id": args.model_id,
        "sample_rate": int(args.sample_rate),
        "window_size_s": float(args.window_size_s),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "windows": str(args.windows),
        "n_windows": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]) if X.ndim == 2 and X.shape[0] else None,
        "elapsed_s": time.perf_counter() - started,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--audio-dir", type=Path, default=Path("data/benchmark/audio"))
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-size-s", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
