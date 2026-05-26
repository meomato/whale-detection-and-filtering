"""Extract frozen BEATs embeddings through the local Voxaboxen checkout.

Run this script with the Voxaboxen Python environment, for example:

    ../voxaboxen/.venv/Scripts/python.exe filtering/benchmark/extract_voxaboxen_beats.py ...

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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from filtering.benchmark.audio_paths import resolve_audio_path


def _load_encoder(voxaboxen_dir: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(voxaboxen_dir))
    from voxaboxen.model.encoders import BEATsEncoder

    args = SimpleNamespace(beats_checkpoint_fp=str(checkpoint), sr=16000)
    encoder = BEATsEncoder(args)
    encoder.freeze()
    encoder.to(device)
    encoder.eval()
    return encoder


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


def extract(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    encoder = _load_encoder(args.voxaboxen_dir, args.beats_checkpoint, device)

    windows = pd.read_csv(args.windows)
    rows: list[dict[str, str | int | float]] = []
    vectors: list[np.ndarray] = []
    batch_audio: list[torch.Tensor] = []
    batch_rows: list[dict[str, str | int | float]] = []

    def flush() -> None:
        if not batch_audio:
            return
        x = torch.stack(batch_audio).to(device)
        with torch.inference_mode():
            feats = encoder(x)
            pooled = feats.mean(dim=1).detach().cpu().numpy().astype(np.float32)
        vectors.extend([pooled[i] for i in range(pooled.shape[0])])
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
        "model_name": "voxaboxen_beats",
        "voxaboxen_dir": str(args.voxaboxen_dir),
        "checkpoint": str(args.beats_checkpoint),
        "windows": str(args.windows),
        "sample_rate": int(args.sample_rate),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "n_windows": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]) if X.ndim == 2 and X.shape[0] else None,
        "elapsed_s": time.perf_counter() - started,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxaboxen-dir", type=Path, default=Path("../voxaboxen"))
    parser.add_argument(
        "--beats-checkpoint",
        type=Path,
        default=Path("../voxaboxen/weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("data/benchmark/audio"))
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-size-s", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
