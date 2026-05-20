"""Summarize the full file-level CV benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


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
METRICS = {
    "average_precision": "Window AP",
    "f1_sound": "F1",
    "precision_sound": "Precision",
    "recall_sound": "Recall",
    "false_positive_rate": "FPR",
    "false_negative_rate": "FNR",
    "recall_at_precision_0.8": "Recall @ P>=0.8",
    "event_f1": "Event F1",
    "event_false_alarm_rate_per_hour": "Event FAR/hour",
    "voxaboxen_style_event_ap_0.5": "Event AP@0.5",
    "voxaboxen_style_event_ap_0.8": "Event AP@0.8",
}
MODEL_ORDER = ["perch_v2", "voxaboxen_beats", "animal2vec_pretrained_meerkat", "wav2vec2_base"]
SCENARIO_ORDER = ["annotations_all", "annotations_v1", "annotations_v2"]
COLORS = {
    "perch_v2": "#2A9D8F",
    "voxaboxen_beats": "#E76F51",
    "animal2vec_pretrained_meerkat": "#5E60CE",
    "wav2vec2_base": "#D9A441",
}


def _read_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test", {})
    detection = data.get("detection", {})
    return {
        "model": data.get("model_name"),
        "model_label": MODEL_LABELS.get(data.get("model_name"), data.get("model_name")),
        "scenario": data.get("scenario"),
        "scenario_label": SCENARIO_LABELS.get(data.get("scenario"), data.get("scenario")),
        "fold": path.parent.name,
        "best_epoch": data.get("best_epoch"),
        "epochs_completed": data.get("epochs_completed"),
        "elapsed_s": data.get("elapsed_s"),
        "average_precision": test.get("average_precision"),
        "f1_sound": test.get("f1_sound"),
        "precision_sound": test.get("precision_sound"),
        "recall_sound": test.get("recall_sound"),
        "false_positive_rate": test.get("false_positive_rate"),
        "false_negative_rate": test.get("false_negative_rate"),
        "recall_at_precision_0.8": test.get("recall_at_precision_0.8"),
        "event_f1": detection.get("event_f1"),
        "event_false_alarm_rate_per_hour": detection.get("event_false_alarm_rate_per_hour"),
        "voxaboxen_style_event_ap_0.5": detection.get("voxaboxen_style_event_ap_0.5"),
        "voxaboxen_style_event_ap_0.8": detection.get("voxaboxen_style_event_ap_0.8"),
        "metrics_path": str(path),
    }


def _summary_table(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, scenario), group in folds.groupby(["model", "scenario"], sort=False):
        row = {
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
            "folds": int(len(group)),
            "epochs_completed_mean": group["epochs_completed"].astype(float).mean(),
            "best_epoch_mean": group["best_epoch"].astype(float).mean(),
        }
        for metric, label in METRICS.items():
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
            row[f"{label} mean"] = row[f"{metric}_mean"]
            row[f"{label} std"] = row[f"{metric}_std"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out["scenario"] = pd.Categorical(out["scenario"], SCENARIO_ORDER, ordered=True)
    return out.sort_values(["scenario", "model"]).reset_index(drop=True)


def _plot_model_metric_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    plot = summary.copy()
    plot["run"] = plot["model_label"].astype(str) + "\n" + plot["scenario_label"].astype(str)
    plot["scenario"] = pd.Categorical(plot["scenario"].astype(str), SCENARIO_ORDER, ordered=True)
    plot["model"] = pd.Categorical(plot["model"].astype(str), MODEL_ORDER, ordered=True)
    plot = plot.sort_values(["scenario", "model"])

    metrics = [
        ("f1_sound_mean", "F1", "#0F4C5C"),
        ("recall_sound_mean", "Recall", "#2A9D8F"),
        ("precision_sound_mean", "Precision", "#D97757"),
        ("average_precision_mean", "Window AP", "#E9C46A"),
    ]

    fig, ax = plt.subplots(figsize=(13.2, 6.1), facecolor="#FBFCFC")
    ax.set_facecolor("#FBFCFC")
    x = list(range(len(plot)))
    width = 0.18
    for idx, (col, label, color) in enumerate(metrics):
        offsets = [pos + (idx - 1.5) * width for pos in x]
        ax.bar(offsets, plot[col].astype(float), width=width, color=color, alpha=0.9, label=label)

    for boundary in [3.5, 7.5]:
        ax.axvline(boundary, color="#C9D5D9", lw=0.9)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, plot["run"], rotation=35, ha="right")
    ax.set_ylabel("CV mean score")
    ax.set_title("Whale sound/noise CV benchmark", pad=12, color="#24343A")
    ax.grid(axis="y", color="#DCE6E8", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_score_heatmap(summary: pd.DataFrame, out_path: Path) -> None:
    work = summary.copy()
    # A compact benchmark score: reward ranking quality, F1 and high-precision recall,
    # penalize false positives.
    work["benchmark_score"] = (
        0.35 * work["average_precision_mean"].astype(float)
        + 0.30 * work["f1_sound_mean"].astype(float)
        + 0.20 * work["recall_at_precision_0.8_mean"].astype(float)
        + 0.15 * (1.0 - work["false_positive_rate_mean"].astype(float))
    )
    work["row"] = work["model_label"].astype(str)
    work["col"] = work["scenario_label"].astype(str)
    rows = [MODEL_LABELS[m] for m in MODEL_ORDER]
    cols = [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
    pivot = work.pivot(index="row", columns="col", values="benchmark_score").reindex(index=rows, columns=cols)

    fig, ax = plt.subplots(figsize=(7.8, 4.8), facecolor="#FBFCFC")
    ax.set_facecolor("#FBFCFC")
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols)
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_title("CV benchmark score", pad=12, color="#24343A")
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            val = pivot.loc[row, col]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="#17343A", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.set_label("higher is better")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _load_cv_predictions(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_dir.glob("*/*/fold_*/test_predictions.csv")):
        model = path.parent.parent.parent.name
        scenario = path.parent.parent.name
        fold = path.parent.name
        table = pd.read_csv(path)
        table["model"] = model
        table["scenario"] = scenario
        table["fold"] = fold
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _plot_cv_pr_curves(predictions: pd.DataFrame, out_path: Path) -> None:
    if predictions.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True, facecolor="#FBFCFC")
    for ax, scenario in zip(axes, SCENARIO_ORDER, strict=False):
        ax.set_facecolor("#FBFCFC")
        scenario_predictions = predictions[predictions["scenario"].astype(str).eq(scenario)]
        for model in MODEL_ORDER:
            data = scenario_predictions[scenario_predictions["model"].astype(str).eq(model)]
            if data.empty or data["label"].nunique() < 2:
                continue
            y_true = data["label"].astype(int).to_numpy()
            scores = data["score_sound"].astype(float).to_numpy()
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax.plot(recall, precision, color=COLORS[model], lw=1.8, label=f"{MODEL_LABELS[model]} AP={ap:.2f}")
        ax.set_title(SCENARIO_LABELS[scenario], color="#24343A", pad=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#DCE6E8", linewidth=0.7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("precision")
    axes[1].set_xlabel("recall")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("CV precision-recall curves", y=1.03, color="#24343A")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_cv_roc_curves(predictions: pd.DataFrame, out_path: Path) -> None:
    if predictions.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True, facecolor="#FBFCFC")
    for ax, scenario in zip(axes, SCENARIO_ORDER, strict=False):
        ax.set_facecolor("#FBFCFC")
        scenario_predictions = predictions[predictions["scenario"].astype(str).eq(scenario)]
        ax.plot([0, 1], [0, 1], color="#AAB7BD", lw=1.0, ls="--")
        for model in MODEL_ORDER:
            data = scenario_predictions[scenario_predictions["model"].astype(str).eq(model)]
            if data.empty or data["label"].nunique() < 2:
                continue
            y_true = data["label"].astype(int).to_numpy()
            scores = data["score_sound"].astype(float).to_numpy()
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, color=COLORS[model], lw=1.8, label=f"{MODEL_LABELS[model]} AUC={auc:.2f}")
        ax.set_title(SCENARIO_LABELS[scenario], color="#24343A", pad=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#DCE6E8", linewidth=0.7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("true positive rate")
    axes[1].set_xlabel("false positive rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("CV ROC curves", y=1.03, color="#24343A")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize(runs_dir: Path, output_dir: Path) -> None:
    rows = [_read_metrics(path) for path in sorted(runs_dir.glob("*/*/fold_*/metrics.json"))]
    if not rows:
        raise FileNotFoundError(f"No CV metrics found under {runs_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    summary = _summary_table(folds)
    summary.to_csv(output_dir / "cv_summary.csv", index=False)
    predictions = _load_cv_predictions(runs_dir)
    _plot_model_metric_comparison(summary, output_dir / "00_model_metric_comparison.png")
    _plot_score_heatmap(summary, output_dir / "01_benchmark_score_heatmap.png")
    _plot_cv_pr_curves(predictions, output_dir / "02_cv_pr_curves.png")
    _plot_cv_roc_curves(predictions, output_dir / "03_cv_roc_curves.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize(args.runs_dir, args.output_dir)
