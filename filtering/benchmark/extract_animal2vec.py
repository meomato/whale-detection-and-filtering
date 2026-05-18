"""Extract frozen animal2vec embeddings through the official local checkout.

This wrapper keeps the official animal2vec repository untouched. Run it with
the animal2vec WSL environment from the repository root, for example:

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    python filtering/benchmark/extract_animal2vec.py \
      --animal2vec-dir ../animal2vec \
      --checkpoint ../animal2vec/checkpoints/animal2vec_large_pretrained_MeerKAT_240507.pt \
      --audio-dir data/benchmark/audio \
      --windows outputs/benchmark/annotations_all/windows.csv \
      --output-dir outputs/benchmark/embeddings/animal2vec_pretrained_meerkat_all

The output format is shared with ``train_downstream.py``:
``embeddings.npy`` plus ``manifest.csv`` with filename/start/end columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF


def _load_model(animal2vec_dir: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(animal2vec_dir))
    import nn  # noqa: F401  # registers animal2vec fairseq models/tasks
    from fairseq import checkpoint_utils

    models, _ = checkpoint_utils.load_model_ensemble([str(checkpoint)])
    model = models[0].to(device)
    model.eval()
    return model


def _read_resampled_file(audio_dir: Path, filename: str, target_sr: int) -> torch.Tensor:
    data, sr = sf.read(str(audio_dir / filename), dtype="float32", always_2d=True)
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
) -> torch.Tensor:
    expected = int(round(fixed_duration_s * target_sr))
    if audio.numel() == 0:
        return torch.zeros(expected, dtype=torch.float32)
    start_frame = max(0, int(round(start_s * target_sr)))
    stop_frame = start_frame + expected
    mono = audio[start_frame:stop_frame]
    if mono.numel() < expected:
        mono = torch.nn.functional.pad(mono, (0, expected - mono.numel()))
    elif mono.numel() > expected:
        mono = mono[:expected]
    return mono


def _pool_layer_results(layer_results: list[torch.Tensor | tuple], top_k_layers: int) -> torch.Tensor:
    selected = layer_results[-top_k_layers:] if top_k_layers > 0 else layer_results
    tensors: list[torch.Tensor] = []
    for layer in selected:
        tensor = layer[0] if isinstance(layer, tuple) else layer
        if tensor.dim() != 3:
            raise ValueError(f"Expected layer result [B, T, C], got {tuple(tensor.shape)}")
        tensors.append(tensor.float())
    stacked = torch.stack(tensors, dim=0).mean(dim=0)
    return stacked.mean(dim=1)


def _normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.nn.functional.layer_norm(x, x.shape) for x in batch])


def extract(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = _load_model(args.animal2vec_dir, args.checkpoint, device)
    windows = pd.read_csv(args.windows)

    rows: list[dict[str, str | int | float]] = []
    vectors: list[np.ndarray] = []
    batch_audio: list[torch.Tensor] = []
    batch_rows: list[dict[str, str | int | float]] = []

    def flush() -> None:
        if not batch_audio:
            return
        x = torch.stack(batch_audio).to(device)
        if args.normalize:
            x = _normalize_batch(x)
        with torch.inference_mode():
            features = model.extract_features(source=x)
            pooled = _pool_layer_results(features["layer_results"], args.average_top_k_layers)
            pooled_np = pooled.detach().cpu().numpy().astype(np.float32)
        vectors.extend([pooled_np[i] for i in range(pooled_np.shape[0])])
        rows.extend(batch_rows)
        batch_audio.clear()
        batch_rows.clear()

    if args.start_index:
        windows = windows.iloc[int(args.start_index) :].reset_index(drop=True)

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
                            "total_in_this_run": total,
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
    np.save(args.output_dir / "embeddings.npy", X)
    with (args.output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "filename", "start_s", "end_s"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "model_name": args.model_name,
        "animal2vec_dir": str(args.animal2vec_dir),
        "checkpoint": str(args.checkpoint),
        "windows": str(args.windows),
        "sample_rate": int(args.sample_rate),
        "window_size_s": float(args.window_size_s),
        "batch_size": int(args.batch_size),
        "average_top_k_layers": int(args.average_top_k_layers),
        "normalize": bool(args.normalize),
        "device": str(device),
        "n_windows": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]) if X.ndim == 2 and X.shape[0] else None,
        "elapsed_s": time.perf_counter() - started,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal2vec-dir", type=Path, default=Path("../animal2vec"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, default=Path("data/benchmark/audio"))
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="animal2vec_pretrained_meerkat")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--window-size-s", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--average-top-k-layers", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
