"""Create benchmark windows from prepared annotation and split manifests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf

from filtering.benchmark.audio_paths import resolve_audio_path


SCENARIOS = ("annotations_all", "annotations_v1", "annotations_v2")


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start_s, end_s in sorted(intervals):
        if end_s <= start_s:
            continue
        if not merged or start_s > merged[-1][1]:
            merged.append((start_s, end_s))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_s))
    return merged


def _label_window(
    start_s: float,
    end_s: float,
    sound_events: list[tuple[float, float]],
    min_overlap_s: float,
) -> str:
    for event_start, event_end in sound_events:
        overlap = min(end_s, event_end) - max(start_s, event_start)
        if overlap >= min_overlap_s:
            return "sound"
    return "noise"


def _load_sound_events(path: Path) -> dict[str, list[tuple[float, float]]]:
    table = pd.read_csv(path)
    events: dict[str, list[tuple[float, float]]] = {}
    if table.empty:
        return events
    table = table[table["label"].astype(str).str.lower() == "sound"]
    for filename, group in table.groupby("audio", sort=False):
        intervals = [(float(row.start_s), float(row.end_s)) for row in group.itertuples()]
        events[str(filename)] = _merge_intervals(intervals)
    return events


def _write_windows(
    audio_dir: Path,
    split_table: pd.DataFrame,
    sound_events_by_file: dict[str, list[tuple[float, float]]],
    output_path: Path,
    window_size_s: float,
    hop_size_s: float,
    min_overlap_s: float,
) -> dict[str, int]:
    rows: list[dict[str, str | int | float]] = []
    label_counts: Counter[str] = Counter()
    for split_row in split_table.itertuples(index=False):
        filename = str(split_row.filename)
        audio_path = resolve_audio_path(audio_dir, filename)
        info = sf.info(str(audio_path))
        duration_s = float(info.frames / info.samplerate)
        sound_events = sound_events_by_file.get(filename, [])
        start_s = 0.0
        window_index = 0
        while start_s < duration_s:
            end_s = start_s + window_size_s
            label = _label_window(start_s, end_s, sound_events, min_overlap_s)
            label_counts[label] += 1
            rows.append(
                {
                    "row_index": len(rows),
                    "filename": filename,
                    "window_index": window_index,
                    "start_s": round(start_s, 6),
                    "end_s": round(end_s, 6),
                    "label": label,
                    "split": str(split_row.split),
                    "sample_rate": int(info.samplerate),
                    "duration_s": round(duration_s, 6),
                }
            )
            window_index += 1
            start_s += hop_size_s

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "filename",
                "window_index",
                "start_s",
                "end_s",
                "label",
                "split",
                "sample_rate",
                "duration_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return dict(label_counts)


def prepare(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "audio_dir": str(args.audio_dir),
        "source_root": str(args.source_root),
        "window_size_s": float(args.window_size_s),
        "hop_size_s": float(args.hop_size_s),
        "min_sound_overlap_s": float(args.min_sound_overlap_s),
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        src_dir = args.source_root / scenario
        out_dir = args.output_root / scenario
        annotations_path = src_dir / "annotations_manifest.csv"
        splits_path = src_dir / "splits.csv"
        if not annotations_path.is_file() or not splits_path.is_file():
            raise FileNotFoundError(f"Missing prepared manifests for {scenario}: {src_dir}")

        sound_events = _load_sound_events(annotations_path)
        split_table = pd.read_csv(splits_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        split_table.to_csv(out_dir / "splits.csv", index=False)
        pd.read_csv(annotations_path).to_csv(out_dir / "annotations_manifest.csv", index=False)
        label_counts = _write_windows(
            audio_dir=args.audio_dir,
            split_table=split_table,
            sound_events_by_file=sound_events,
            output_path=out_dir / "windows.csv",
            window_size_s=float(args.window_size_s),
            hop_size_s=float(args.hop_size_s),
            min_overlap_s=float(args.min_sound_overlap_s),
        )
        metadata = {
            "scenario": scenario,
            "window_size_s": float(args.window_size_s),
            "hop_size_s": float(args.hop_size_s),
            "min_sound_overlap_s": float(args.min_sound_overlap_s),
            "window_label_counts": label_counts,
            "source_manifests": str(src_dir),
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary["scenarios"][scenario] = metadata
        print(f"{scenario}: labels={label_counts}")

    (args.output_root / "window_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/orcasound/audio"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/benchmark_context5_hop1"))
    parser.add_argument("--window-size-s", type=float, default=5.0)
    parser.add_argument("--hop-size-s", type=float, default=1.0)
    parser.add_argument("--min-sound-overlap-s", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
