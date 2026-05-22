"""Build the sound/noise manifests used by the benchmark.

Inputs are Label Studio JSON exports plus the configured audio folder. The
outputs are shared window manifests and Voxaboxen-ready Raven selection tables.
Source audio stays outside the repository.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import soundfile as sf
from omegaconf import DictConfig

from filtering.benchmark.audio_paths import AUDIO_SUFFIXES, resolve_audio_path


SCENARIOS = ("annotations_all", "annotations_v1", "annotations_v2")
ARTIFACT_LABELS = {"artifact", "artifacts"}
ANNOTATION2_TRAIN_IDS = {1, 10, 13, 15, 16, 17, 19, 20, 23, 25, 26, 33, 46, 57, 60}
ANNOTATION2_TEST_IDS = {9, 11, 12, 14, 18, 21, 22, 24, 28, 32}


@dataclass(frozen=True)
class Event:
    filename: str
    start_s: float
    end_s: float
    label: str
    source: str
    item_id: int | None


@dataclass(frozen=True)
class Item:
    filename: str
    source: str
    item_id: int | None
    events: tuple[Event, ...]


def _labels(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None and str(x)]
    return [str(raw)]


def _audio_name(item: dict[str, Any]) -> str | None:
    candidates: list[Any] = []
    data = item.get("data")
    if isinstance(data, dict):
        candidates.append(data.get("audio"))
    candidates.extend([item.get("audio"), item.get("file_upload")])
    for raw in candidates:
        if raw:
            return Path(str(raw).replace("\\", "/")).name
    return None


def _raw_events(item: dict[str, Any]) -> list[dict[str, Any]]:
    direct = item.get("label")
    if isinstance(direct, list):
        return [x for x in direct if isinstance(x, dict)]

    events: list[dict[str, Any]] = []
    for annotation in item.get("annotations", []) or []:
        if not isinstance(annotation, dict) or annotation.get("was_cancelled"):
            continue
        for result in annotation.get("result", []) or []:
            value = result.get("value", {}) if isinstance(result, dict) else {}
            if isinstance(value, dict) and "start" in value and "end" in value:
                events.append(value)
    return events


def _binary_label(labels: list[str]) -> str:
    lowered = {x.lower().strip() for x in labels}
    if lowered and lowered.issubset(ARTIFACT_LABELS):
        return "artifact"
    return "sound"


def _load_items(path: Path, source: str) -> list[Item]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON array in {path}")

    items: list[Item] = []
    for obj in payload:
        if not isinstance(obj, dict):
            continue
        filename = _audio_name(obj)
        if not filename:
            continue
        item_id_raw = obj.get("id") or obj.get("annotation_id")
        item_id = int(item_id_raw) if item_id_raw is not None else None
        events: list[Event] = []
        for raw in _raw_events(obj):
            start_s = float(raw.get("start", 0.0))
            end_s = float(raw.get("end", 0.0))
            if end_s <= start_s:
                continue
            labels = _labels(raw.get("labels") or raw.get("choices"))
            events.append(
                Event(
                    filename=filename,
                    start_s=start_s,
                    end_s=end_s,
                    label=_binary_label(labels),
                    source=source,
                    item_id=item_id,
                )
            )
        items.append(Item(filename=filename, source=source, item_id=item_id, events=tuple(events)))
    return items


def _merge_sound_events(events: list[Event]) -> list[Event]:
    sound_events = [e for e in events if e.label == "sound"]
    if not sound_events:
        return []
    first = sound_events[0]
    intervals = sorted((e.start_s, e.end_s) for e in sound_events)
    merged: list[tuple[float, float]] = []
    for start_s, end_s in intervals:
        if not merged or start_s > merged[-1][1]:
            merged.append((start_s, end_s))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_s))
    return [
        Event(first.filename, start_s, end_s, "sound", first.source, first.item_id)
        for start_s, end_s in merged
    ]


def _scenario_items(items_by_source: dict[str, list[Item]], scenario: str) -> list[Item]:
    if scenario == "annotations_v1":
        return items_by_source["annotations_v1"]
    if scenario == "annotations_v2":
        return items_by_source["annotations_v2"]
    if scenario == "annotations_all":
        return items_by_source["annotations_v1"] + items_by_source["annotations_v2"]
    raise ValueError(f"Unknown scenario: {scenario}")


def _fixed_split_for_annotation2(item: Item) -> str | None:
    if item.item_id in ANNOTATION2_TRAIN_IDS:
        return "train"
    if item.item_id in ANNOTATION2_TEST_IDS:
        return "test"
    return None


def _split_items(items: list[Item], scenario: str, seed: int, val_fraction: float) -> dict[str, str]:
    split_by_file: dict[str, str] = {}
    train_candidates: set[str] = set()
    unsplit_candidates: set[str] = set()

    for item in items:
        fixed = _fixed_split_for_annotation2(item) if item.source == "annotations_v2" else None
        if fixed == "test":
            split_by_file[item.filename] = "test"
        elif fixed == "train":
            train_candidates.add(item.filename)
        else:
            unsplit_candidates.add(item.filename)

    rng = random.Random(seed)
    if scenario != "annotations_v2":
        pool = sorted(f for f in unsplit_candidates if f not in split_by_file)
        rng.shuffle(pool)
        n_test = max(1, round(len(pool) * 0.25)) if pool else 0
        for filename in pool[:n_test]:
            split_by_file[filename] = "test"
        train_candidates.update(pool[n_test:])

    train_pool = sorted(f for f in train_candidates if f not in split_by_file)
    rng.shuffle(train_pool)
    n_val = max(1, round(len(train_pool) * val_fraction)) if len(train_pool) >= 3 else 0
    for filename in train_pool[:n_val]:
        split_by_file[filename] = "val"
    for filename in train_pool[n_val:]:
        split_by_file[filename] = "train"
    return split_by_file


def _events_by_file(items: list[Item]) -> dict[str, list[Event]]:
    grouped: dict[str, list[Event]] = {}
    for item in items:
        grouped.setdefault(item.filename, []).extend(item.events)
    return grouped


def _write_annotations_manifest(items: list[Item], path: Path) -> None:
    rows: list[dict[str, str | float]] = []
    for filename, events in sorted(_events_by_file(items).items()):
        for event in _merge_sound_events(events):
            rows.append(
                {
                    "audio": filename,
                    "start_s": event.start_s,
                    "end_s": event.end_s,
                    "label": "sound",
                }
            )
        for event in events:
            if event.label == "artifact":
                rows.append(
                    {
                        "audio": filename,
                        "start_s": event.start_s,
                        "end_s": event.end_s,
                        "label": "artifact",
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "start_s", "end_s", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _window_label(
    start_s: float,
    end_s: float,
    sound_events: list[Event],
    min_overlap_s: float,
) -> str:
    for event in sound_events:
        overlap = min(end_s, event.end_s) - max(start_s, event.start_s)
        if overlap >= min_overlap_s:
            return "sound"
    return "noise"


def _write_window_manifest(
    items: list[Item],
    split_by_file: dict[str, str],
    audio_dir: Path,
    path: Path,
    window_size_s: float,
    hop_size_s: float,
    min_overlap_s: float,
) -> dict[str, int]:
    rows: list[dict[str, str | int | float]] = []
    label_counts = {"sound": 0, "noise": 0}
    events_by_file = _events_by_file(items)
    for row_index, filename in enumerate(sorted(events_by_file)):
        audio_path = resolve_audio_path(audio_dir, filename)
        info = sf.info(str(audio_path))
        duration_s = float(info.frames / info.samplerate)
        sound_events = _merge_sound_events(events_by_file[filename])
        start_s = 0.0
        local_index = 0
        while start_s < duration_s:
            end_s = min(duration_s, start_s + window_size_s)
            label = _window_label(start_s, end_s, sound_events, min_overlap_s)
            label_counts[label] += 1
            rows.append(
                {
                    "row_index": len(rows),
                    "filename": filename,
                    "window_index": local_index,
                    "start_s": round(start_s, 6),
                    "end_s": round(end_s, 6),
                    "label": label,
                    "split": split_by_file[filename],
                    "sample_rate": info.samplerate,
                    "duration_s": round(duration_s, 6),
                }
            )
            local_index += 1
            start_s += hop_size_s
    with path.open("w", encoding="utf-8", newline="") as f:
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
    return label_counts


def _write_split_manifest(items: list[Item], split_by_file: dict[str, str], path: Path) -> None:
    rows: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for item in items:
        if item.filename in seen:
            continue
        seen.add(item.filename)
        rows.append(
            {
                "filename": item.filename,
                "split": split_by_file[item.filename],
                "source": item.source,
                "item_id": "" if item.item_id is None else item.item_id,
            }
        )
    rows.sort(key=lambda r: (str(r["split"]), str(r["source"]), str(r["filename"])))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "split", "source", "item_id"])
        writer.writeheader()
        writer.writerows(rows)


def _write_selection_table(path: Path, events: list[Event]) -> None:
    rows = [
        {"Begin Time (s)": e.start_s, "End Time (s)": e.end_s, "Annotation": "sound"}
        for e in _merge_sound_events(events)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Begin Time (s)", "End Time (s)", "Annotation"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_voxaboxen_dataset(
    items: list[Item],
    split_by_file: dict[str, str],
    audio_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, str]]] = {
        "train": [],
        "val": [],
        "test": [],
        "all": [],
    }
    map_rows: list[dict[str, str]] = []
    for idx, (filename, events) in enumerate(sorted(_events_by_file(items).items())):
        short_fn = f"vx_{idx:04d}"
        table_path = (out_dir / "selection_tables" / f"{short_fn}.txt").resolve()
        _write_selection_table(table_path, events)
        audio_path = resolve_audio_path(audio_dir, filename)
        row = {
            "fn": short_fn,
            "audio_fp": str(audio_path.resolve()),
            "selection_table_fp": str(table_path),
        }
        map_rows.append(
            {
                "fn": short_fn,
                "filename": filename,
                "audio_fp": row["audio_fp"],
                "split": split_by_file[filename],
            }
        )
        rows_by_split["all"].append(row)
        rows_by_split[split_by_file[filename]].append(row)

    for split, rows in rows_by_split.items():
        with (out_dir / f"{split}_info.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["fn", "audio_fp", "selection_table_fp"])
            writer.writeheader()
            writer.writerows(rows)

    with (out_dir / "file_map.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fn", "filename", "audio_fp", "split"])
        writer.writeheader()
        writer.writerows(map_rows)

    project_dir = out_dir.parent / "voxaboxen_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    project_config = "\n".join(
        [
            f"data_dir: {out_dir.resolve()}",
            f"project_dir: {project_dir.resolve()}",
            f"train_info_fp: {(out_dir / 'train_info.csv').resolve()}",
            f"val_info_fp: {(out_dir / 'val_info.csv').resolve()}",
            f"test_info_fp: {(out_dir / 'test_info.csv').resolve()}",
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


def _write_metadata(
    path: Path,
    scenario: str,
    items: list[Item],
    split_by_file: dict[str, str],
    label_counts: dict[str, int],
    cfg: DictConfig,
) -> None:
    split_counts = {name: list(split_by_file.values()).count(name) for name in ("train", "val", "test")}
    ann2_ids = {item.item_id for item in items if item.source == "annotations_v2" and item.item_id is not None}
    metadata = {
        "scenario": scenario,
        "audio_dir": str(cfg.audio_dir),
        "annotation1_json": str(cfg.annotation1_json),
        "annotation2_json": str(cfg.annotation2_json),
        "n_files": len(split_by_file),
        "split_counts": split_counts,
        "window_size_s": float(cfg.window_size_s),
        "hop_size_s": float(cfg.hop_size_s),
        "min_sound_overlap_s": float(cfg.min_sound_overlap_s),
        "window_label_counts": label_counts,
        "annotation2_requested_train_ids": sorted(ANNOTATION2_TRAIN_IDS),
        "annotation2_requested_test_ids": sorted(ANNOTATION2_TEST_IDS),
        "annotation2_missing_train_ids": sorted(ANNOTATION2_TRAIN_IDS - ann2_ids),
        "annotation2_missing_test_ids": sorted(ANNOTATION2_TEST_IDS - ann2_ids),
        "label_policy": {
            "sound": "any non-artifact whale label, including clicks/vocalizations",
            "noise": "windows without sufficient sound overlap",
            "artifact": "kept in annotations_manifest, treated as non-sound/noise downstream",
        },
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_corpus_summary(audio_dir: Path, out_root: Path) -> None:
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    total_s = 0.0
    files = sorted(path for suffix in AUDIO_SUFFIXES for path in audio_dir.glob(f"*{suffix}"))
    for path in files:
        info = sf.info(str(path))
        sample_rates[int(info.samplerate)] += 1
        channels[int(info.channels)] += 1
        total_s += float(info.frames / info.samplerate)
    summary = {
        "audio_dir": str(audio_dir),
        "audio_files": len(files),
        "total_hours": total_s / 3600.0,
        "sample_rates": {str(k): v for k, v in sorted(sample_rates.items())},
        "channels": {str(k): v for k, v in sorted(channels.items())},
    }
    (out_root / "corpus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config.benchmark
    audio_dir = Path(hydra.utils.to_absolute_path(str(cfg.audio_dir)))
    out_root = Path(hydra.utils.to_absolute_path(str(cfg.output_dir)))
    items_by_source = {
        "annotations_v1": _load_items(
            Path(hydra.utils.to_absolute_path(str(cfg.annotation1_json))),
            "annotations_v1",
        ),
        "annotations_v2": _load_items(
            Path(hydra.utils.to_absolute_path(str(cfg.annotation2_json))),
            "annotations_v2",
        ),
    }

    for scenario in SCENARIOS:
        items = _scenario_items(items_by_source, scenario)
        missing = []
        for item in items:
            try:
                resolve_audio_path(audio_dir, item.filename)
            except FileNotFoundError:
                missing.append(item.filename)
        missing = sorted(set(missing))
        if missing:
            raise FileNotFoundError(f"{scenario}: missing audio files: {missing[:10]}")

        scenario_dir = out_root / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        split_by_file = _split_items(
            items,
            scenario=scenario,
            seed=int(cfg.seed),
            val_fraction=float(cfg.val_fraction),
        )
        _write_annotations_manifest(items, scenario_dir / "annotations_manifest.csv")
        _write_split_manifest(items, split_by_file, scenario_dir / "splits.csv")
        label_counts = _write_window_manifest(
            items,
            split_by_file,
            audio_dir,
            scenario_dir / "windows.csv",
            window_size_s=float(cfg.window_size_s),
            hop_size_s=float(cfg.hop_size_s),
            min_overlap_s=float(cfg.min_sound_overlap_s),
        )
        _write_voxaboxen_dataset(
            items,
            split_by_file,
            audio_dir,
            scenario_dir / "voxaboxen_dataset",
        )
        _write_metadata(scenario_dir / "metadata.json", scenario, items, split_by_file, label_counts, cfg)

        print(f"{scenario}: files={len(split_by_file)} labels={label_counts}")
        print(f"  {scenario_dir}")
    _write_corpus_summary(audio_dir, out_root)


if __name__ == "__main__":
    main()
