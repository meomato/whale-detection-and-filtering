"""Make the small summary tables and plots for the model check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SKIP_MODELS = {"animal2vec_local"}
SCENARIO_ORDER = ["annotations_all", "annotations_v1", "annotations_v2"]
PLOT_COLORS = ["#3A6EA5", "#E07A5F", "#59A14F", "#F2CC8F"]
MODEL_LABELS = {
    "animal2vec_pretrained_meerkat": "animal2vec",
    "perch_v2": "Perch 2.0",
    "voxaboxen_beats": "Voxaboxen",
    "wav2vec2_base": "Wav2Vec2",
}
SCENARIO_LABELS = {
    "annotations_all": "all labels",
    "annotations_v1": "labels v1",
    "annotations_v2": "labels v2",
}


def _flatten_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test", {})
    return {
        "model": data.get("model_name"),
        "scenario": data.get("scenario"),
        "best_epoch": data.get("best_epoch"),
        "epochs_completed": data.get("epochs_completed"),
        "elapsed_s": data.get("elapsed_s"),
        "accuracy": test.get("accuracy"),
        "precision_sound": test.get("precision_sound"),
        "recall_sound": test.get("recall_sound"),
        "f1_sound": test.get("f1_sound"),
        "macro_f1": test.get("macro_f1"),
        "roc_auc": test.get("roc_auc"),
        "average_precision": test.get("average_precision"),
        "false_positive_rate": test.get("false_positive_rate"),
        "false_negative_rate": test.get("false_negative_rate"),
        "metrics_path": str(path),
    }


def summarize(root: Path, output_dir: Path, skip_models: set[str]) -> None:
    rows = [_flatten_metrics(path) for path in root.glob("**/metrics.json")]
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        (output_dir / "summary.csv").write_text("", encoding="utf-8")
        print(f"No metrics.json files found under {root}")
        return
    df = pd.DataFrame(rows).sort_values(["model", "scenario"])
    if skip_models:
        df = df.loc[~df["model"].isin(skip_models)].copy()
    df["from_runs_dir"] = df["metrics_path"].str.replace("\\", "/", regex=False).str.contains("/runs/")
    df = (
        df.sort_values(["model", "scenario", "from_runs_dir"], ascending=[True, True, False])
        .drop_duplicates(["model", "scenario"], keep="first")
        .drop(columns=["from_runs_dir"])
    )
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    df = df.sort_values(["scenario", "model"])
    df.to_csv(output_dir / "summary.csv", index=False)

    pivot = df.pivot_table(index="model", columns="scenario", values="f1_sound", aggfunc="max", observed=False)
    pivot.to_csv(output_dir / "f1_by_model_scenario.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_df = df.assign(
        run=df["model"].map(MODEL_LABELS).fillna(df["model"].astype(str))
        + "\n"
        + df["scenario"].astype(str).map(SCENARIO_LABELS).fillna(df["scenario"].astype(str))
    )
    plot_df.plot.bar(
        x="run",
        y=["f1_sound", "recall_sound", "precision_sound", "average_precision"],
        ax=ax,
        color=PLOT_COLORS,
        width=0.78,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("score")
    ax.set_title("Whale sound/noise benchmark", pad=12)
    ax.legend(["F1", "Recall", "Precision", "PR-AUC"], frameon=False, ncol=4, loc="upper center")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "model_metric_comparison.png", dpi=160)
    plt.close(fig)
    print(f"Wrote {output_dir / 'summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/benchmark/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark/report"))
    parser.add_argument(
        "--include-local-animal2vec",
        action="store_true",
        help="Also include extra animal2vec diagnostic runs in summary.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    skip = set() if args.include_local_animal2vec else DEFAULT_SKIP_MODELS
    summarize(args.root, args.output_dir, skip)
