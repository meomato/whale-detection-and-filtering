"""Prepare ordered long-file windows for inference review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def _has_overlap(
    start_s: float,
    end_s: float,
    events: list[tuple[float, float]],
    min_overlap_s: float,
) -> bool:
    for event_start, event_end in events:
        if event_start >= end_s:
            break
        overlap = min(end_s, event_end) - max(start_s, event_start)
        if overlap >= min_overlap_s:
            return True
    return False


def prepare(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order = pd.read_csv(args.segment_order)
    sound = pd.read_csv(args.sound_events)
    sound_events = sorted(
        (float(row.global_start_s), float(row.global_end_s))
        for row in sound.itertuples(index=False)
        if str(row.label) == "sound"
    )

    rows: list[dict[str, int | float | str]] = []
    for seg in order.itertuples(index=False):
        filename = str(seg.file)
        global_start = float(seg.global_start_s)
        duration = float(seg.duration_s)
        local_start = 0.0
        while local_start < duration:
            local_end = local_start + float(args.window_size_s)
            if duration - local_start < float(args.min_window_s):
                break
            start_global = global_start + local_start
            end_global = global_start + local_end
            label = int(_has_overlap(start_global, end_global, sound_events, float(args.min_sound_overlap_s)))
            rows.append(
                {
                    "row_index": len(rows),
                    "recording_id": "long_file_A",
                    "filename": filename,
                    "segment_index": int(seg.index),
                    "start_s": round(local_start, 6),
                    "end_s": round(local_end, 6),
                    "global_start_s": round(start_global, 6),
                    "global_end_s": round(end_global, 6),
                    "label": label,
                    "label_name": "sound" if label else "noise",
                }
            )
            local_start += float(args.hop_size_s)

    out_path = args.output_dir / "windows.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "windows": len(rows),
        "sound_windows": sum(int(r["label"]) for r in rows),
        "noise_windows": sum(1 - int(r["label"]) for r in rows),
    }
    (args.output_dir / "windows_summary.json").write_text(
        "{\n"
        + ",\n".join(f'  "{key}": {value}' for key, value in summary.items())
        + "\n}\n",
        encoding="utf-8",
    )
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment-order",
        type=Path,
        default=Path("data/long_file_A/orcasound_2020-06-26-SRKW-Lpod/label_studio_segments_ordered.csv"),
    )
    parser.add_argument(
        "--sound-events",
        type=Path,
        default=Path(
            "data/long_file_A/orcasound_2020-06-26-SRKW-Lpod/annotations/annotation3_long_A_sound_events_merged.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference_review/orcasound_2020-06-26-SRKW-Lpod"))
    parser.add_argument("--window-size-s", type=float, default=5.0)
    parser.add_argument("--hop-size-s", type=float, default=1.0)
    parser.add_argument("--min-window-s", type=float, default=0.5)
    parser.add_argument("--min-sound-overlap-s", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
