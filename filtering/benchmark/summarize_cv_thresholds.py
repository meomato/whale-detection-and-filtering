"""Summarize CV threshold sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "animal2vec_pretrained_meerkat": "animal2vec pretrained",
    "perch_v2": "Perch 2.0",
    "voxaboxen_beats": "Voxaboxen BEATs",
    "wav2vec2_base": "Wav2Vec2 base",
}
SCENARIO_LABELS = {
    "annotations_all": "all",
    "annotations_v1": "v1",
    "annotations_v2": "v2",
}
MODEL_ORDER = ["perch_v2", "voxaboxen_beats", "animal2vec_pretrained_meerkat", "wav2vec2_base"]
SCENARIO_ORDER = ["annotations_all", "annotations_v1", "annotations_v2"]
MODE_ORDER = ["balanced", "clean"]
METRICS = ["threshold", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate"]


def summarize(thresholds_dir: Path, output_dir: Path) -> None:
    rows = []
    for path in sorted(thresholds_dir.glob("*/*/fold_*/selected_thresholds.csv")):
        model = path.parents[2].name
        scenario = path.parents[1].name
        fold = path.parent.name
        table = pd.read_csv(path)
        table = table[table["split"].eq("test")].copy()
        table["model"] = model
        table["model_label"] = MODEL_LABELS.get(model, model)
        table["scenario"] = scenario
        table["scenario_label"] = SCENARIO_LABELS.get(scenario, scenario)
        table["fold"] = fold
        rows.extend(table.to_dict("records"))
    if not rows:
        raise FileNotFoundError(f"No selected_thresholds.csv files found under {thresholds_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(output_dir / "cv_threshold_fold_metrics.csv", index=False)

    summary_rows = []
    for (model, scenario, mode), group in folds.groupby(["model", "scenario", "mode"], sort=False):
        row = {
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
            "mode": mode,
            "folds": int(len(group)),
        }
        for metric in METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary["scenario"] = pd.Categorical(summary["scenario"], SCENARIO_ORDER, ordered=True)
    summary["mode"] = pd.Categorical(summary["mode"], MODE_ORDER, ordered=True)
    summary = summary.sort_values(["scenario", "model", "mode"]).reset_index(drop=True)
    summary.to_csv(output_dir / "cv_threshold_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize(args.thresholds_dir, args.output_dir)
