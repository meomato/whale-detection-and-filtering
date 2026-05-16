"""Collect the benchmark plots into the git-friendly reports folder."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
from pathlib import Path


PLOT_NAMES = {
    "epoch_metrics.png": "epochs",
    "roc_curve.png": "roc",
    "pr_curve.png": "pr",
    "confusion_matrix.png": "confusion",
}
DEFAULT_SKIP_MODELS = {"animal2vec_local"}


def collect(args: argparse.Namespace) -> None:
    skip_models = set(args.skip_model or [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_files: set[Path] = set()

    def copy_plot(src: Path, dst: Path) -> None:
        if dst.exists():
            os.chmod(dst, stat.S_IWRITE)
        shutil.copy2(src, dst)
        expected_files.add(dst.resolve())

    copied = 0
    report_plot = args.report_dir / "model_metric_comparison.png"
    if report_plot.is_file():
        copy_plot(report_plot, args.output_dir / "00_model_metric_comparison.png")
        copied += 1

    for metrics_path in sorted(args.runs_dir.glob("*/*/metrics.json")):
        scenario_dir = metrics_path.parent
        scenario = scenario_dir.name
        model = scenario_dir.parent.name
        if model in skip_models:
            continue
        for filename, label in PLOT_NAMES.items():
            src = scenario_dir / filename
            if not src.is_file():
                continue
            dst = args.output_dir / f"{model}__{scenario}__{label}.png"
            copy_plot(src, dst)
            copied += 1

    readme = args.output_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Whale sound/noise benchmark figures.",
                "",
                "Open this first:",
                "00_model_metric_comparison.png",
                "",
                "Other files are named like:",
                "<model>__<annotation_set>__<plot_type>.png",
                "",
                "Plot types:",
                "- epochs: validation F1 / precision / recall by epoch",
                "- roc: ROC curve",
                "- pr: precision-recall curve",
                "- confusion: confusion matrix",
                "",
                "Models:",
                "- animal2vec_pretrained_meerkat",
                "- perch_v2",
                "- voxaboxen_beats",
                "- wav2vec2_base",
                "",
                "Annotation sets:",
                "- annotations_all",
                "- annotations_v1",
                "- annotations_v2",
                "",
                f"Plots copied: {copied}",
            ]
        ),
        encoding="utf-8",
    )
    expected_files.add(readme.resolve())

    for old_file in args.output_dir.glob("*.png"):
        if old_file.resolve() not in expected_files:
            os.chmod(old_file, stat.S_IWRITE)
            old_file.unlink()

    print(f"Copied {copied} plots to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/benchmark/runs"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/benchmark/report"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/benchmark_figures"))
    parser.add_argument("--skip-model", action="append", default=sorted(DEFAULT_SKIP_MODELS))
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
