"""Build Voxaboxen training files from benchmark manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from filtering.benchmark.audio_paths import resolve_audio_path


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


def _write_selection_table(path: Path, intervals: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Begin Time (s)", "End Time (s)", "Annotation"],
            delimiter="\t",
        )
        writer.writeheader()
        for start_s, end_s in intervals:
            writer.writerow(
                {
                    "Begin Time (s)": round(float(start_s), 6),
                    "End Time (s)": round(float(end_s), 6),
                    "Annotation": "sound",
                }
            )


def prepare(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    data_dir = out_dir / "data"
    project_dir = out_dir / "project"
    selection_dir = data_dir / "selection_tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(args.annotations)
    annotations = annotations[annotations["label"].astype(str).str.lower() == "sound"].copy()
    splits = pd.read_csv(args.splits)
    split_map = dict(zip(splits["filename"].astype(str), splits["split"].astype(str)))

    rows_by_split = {"train": [], "val": [], "test": [], "all": []}
    map_rows = []
    for idx, filename in enumerate(sorted(split_map)):
        file_events = annotations[annotations["audio"].astype(str) == filename]
        intervals = _merge_intervals(
            [(float(row.start_s), float(row.end_s)) for row in file_events.itertuples()]
        )
        short_name = f"vx_{idx:04d}"
        table_path = selection_dir / f"{short_name}.txt"
        _write_selection_table(table_path, intervals)
        audio_path = resolve_audio_path(args.audio_dir, filename).resolve()
        row = {
            "fn": short_name,
            "audio_fp": str(audio_path),
            "selection_table_fp": str(table_path.resolve()),
        }
        split = split_map[filename]
        if split in rows_by_split:
            rows_by_split[split].append(row)
        rows_by_split["all"].append(row)
        map_rows.append({"fn": short_name, "filename": filename, "split": split, "audio_fp": str(audio_path)})

    for split, rows in rows_by_split.items():
        with (data_dir / f"{split}_info.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["fn", "audio_fp", "selection_table_fp"])
            writer.writeheader()
            writer.writerows(rows)

    with (data_dir / "file_map.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fn", "filename", "split", "audio_fp"])
        writer.writeheader()
        writer.writerows(map_rows)

    project_config = "\n".join(
        [
            f"data_dir: {data_dir.resolve()}",
            f"project_dir: {project_dir.resolve()}",
            f"train_info_fp: {(data_dir / 'train_info.csv').resolve()}",
            f"val_info_fp: {(data_dir / 'val_info.csv').resolve()}",
            f"test_info_fp: {(data_dir / 'test_info.csv').resolve()}",
            "unknown_label: Unknown",
            "label_mapping:",
            "  Unknown: Unknown",
            "  sound: whale_sound",
            "label_set:",
            "- whale_sound",
            "",
        ]
    )
    (project_dir / "project_config.yaml").write_text(project_config, encoding="utf-8")

    print(f"train files: {len(rows_by_split['train'])}")
    print(f"val files: {len(rows_by_split['val'])}")
    print(f"test files: {len(rows_by_split['test'])}")
    print(f"project config: {project_dir / 'project_config.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("data/benchmark_audio"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("outputs/benchmark_context5_hop1/annotations_all/annotations_manifest.csv"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("outputs/benchmark_context5_hop1_cv/annotations_all/fold_01/splits.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/voxaboxen_pilot/fold_01_dataset"))
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())

