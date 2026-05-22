"""Prepare a small FLAC upload folder for HPC training."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf


def _find_audio(source_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for suffix in ("*.wav", "*.WAV", "*.flac", "*.FLAC"):
        for path in source_dir.rglob(suffix):
            files.setdefault(path.name, path)
    return files


def _convert_to_flac(src: Path, dst: Path) -> None:
    data, sample_rate = sf.read(str(src), dtype="float32", always_2d=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), data, sample_rate, format="FLAC", subtype="PCM_16")


def prepare(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)
    audio_dir = output_dir / "data" / "benchmark_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for archive in (args.project_zip, args.metadata_zip):
        if archive.is_file():
            shutil.copy2(archive, output_dir / archive.name)

    windows = pd.read_csv(args.windows)
    filenames = sorted(windows["filename"].astype(str).unique())
    audio_by_name = _find_audio(args.source_audio_dir)

    rows = []
    missing = []
    for filename in filenames:
        src = audio_by_name.get(filename)
        if src is None:
            missing.append(filename)
            continue
        dst = audio_dir / f"{Path(filename).stem}.flac"
        if not dst.is_file():
            _convert_to_flac(src, dst)
        rows.append(
            {
                "filename": filename,
                "source": str(src),
                "upload_relative": str(dst.relative_to(output_dir)).replace("\\", "/"),
                "source_bytes": src.stat().st_size,
                "upload_bytes": dst.stat().st_size,
            }
        )

    with (output_dir / "audio_transfer_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "source", "upload_relative", "source_bytes", "upload_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    readme = """HPC upload folder for whale fine-tuning.

Upload this folder content to /home/lsavchenko/whales/.

On HPC:
cd ~/whales
mkdir -p repo
unzip whale_project_hpc.zip -d repo
unzip whale_training_metadata_hpc.zip -d repo
mkdir -p repo/data
cp -r data/benchmark_audio repo/data/benchmark_audio
"""
    (output_dir / "README_UPLOAD.txt").write_text(readme, encoding="utf-8")

    source_gb = sum(row["source_bytes"] for row in rows) / 1024**3
    upload_gb = sum(row["upload_bytes"] for row in rows) / 1024**3
    print(f"files: {len(rows)}")
    print(f"missing: {len(missing)}")
    print(f"source: {source_gb:.2f} GB")
    print(f"flac: {upload_gb:.2f} GB")
    print(f"output: {output_dir}")
    if missing:
        print("missing files:")
        for filename in missing:
            print(f"  {filename}")
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audio-dir", type=Path, default=Path("D:/data"))
    parser.add_argument("--windows", type=Path, default=Path("outputs/benchmark_context5_hop1/annotations_all/windows.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/hpc_upload_flac"))
    parser.add_argument("--project-zip", type=Path, default=Path("whale_project_hpc.zip"))
    parser.add_argument("--metadata-zip", type=Path, default=Path("whale_training_metadata_hpc.zip"))
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())

