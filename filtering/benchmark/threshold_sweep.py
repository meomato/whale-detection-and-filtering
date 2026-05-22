"""Pick simple decision thresholds on validation scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


COLORS = {
    "f1": "#0F4C5C",
    "precision": "#D97757",
    "recall": "#2A9D8F",
    "threshold": "#6D597A",
    "grid": "#DCE6E8",
}


def _score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    raw = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-raw))


def _metrics_for_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tp = int(((y_true == 1) & (pred == 1)).sum())
    evaluated_seconds = float(len(y_true))
    return {
        "threshold": float(threshold),
        "precision": float(precision[1]),
        "recall": float(recall[1]),
        "f1": float(f1[1]),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predicted_sound_windows": int(pred.sum()),
        "predicted_sound_minutes_per_hour": float(pred.sum() / evaluated_seconds * 60.0)
        if evaluated_seconds > 0
        else 0.0,
    }


def _sweep(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([_metrics_for_threshold(y_true, scores, float(th)) for th in thresholds])


def _pick_thresholds(val_sweep: pd.DataFrame, clean_precision: float) -> dict[str, float]:
    balanced = val_sweep.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    clean_pool = val_sweep[val_sweep["precision"] >= clean_precision]
    if clean_pool.empty:
        clean = val_sweep.sort_values(["precision", "recall", "f1"], ascending=False).iloc[0]
    else:
        clean = clean_pool.sort_values(["recall", "f1", "precision"], ascending=False).iloc[0]
    return {
        "balanced": float(balanced["threshold"]),
        "clean": float(clean["threshold"]),
    }


def _plot_sweep(df: pd.DataFrame, picks: dict[str, float], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.plot(df["threshold"], df["f1"], color=COLORS["f1"], linewidth=2.2, label="F1")
    ax.plot(df["threshold"], df["precision"], color=COLORS["precision"], linewidth=2.0, label="Precision")
    ax.plot(df["threshold"], df["recall"], color=COLORS["recall"], linewidth=2.0, label="Recall")
    for name, threshold in picks.items():
        ax.axvline(threshold, color=COLORS["threshold"], linestyle="--", linewidth=1.2)
        ax.text(threshold, 0.04, name, rotation=90, va="bottom", ha="right", color=COLORS["threshold"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_title(title, pad=10)
    ax.grid(color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor="#FFFFFF")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.head)
    table = pd.read_csv(args.table)
    x = np.load(args.embeddings)
    if len(table) != len(x):
        if not args.embedding_manifest:
            raise ValueError(f"table rows ({len(table)}) != embeddings rows ({len(x)})")
        manifest = pd.read_csv(args.embedding_manifest).reset_index().rename(columns={"index": "_embedding_row"})
        key_cols = ["filename", "start_s", "end_s"]
        merged = table.merge(manifest[key_cols + ["_embedding_row"]], on=key_cols, how="left")
        if merged["_embedding_row"].isna().any():
            missing = int(merged["_embedding_row"].isna().sum())
            raise ValueError(f"could not align {missing} table rows to embedding manifest")
        x = x[merged["_embedding_row"].astype(int).to_numpy()]
        table = merged.drop(columns=["_embedding_row"])
    if len(table) != len(x):
        raise ValueError(f"table rows ({len(table)}) != embeddings rows ({len(x)})")

    x = bundle["scaler"].transform(x)
    scores = _score(bundle["model"], x)
    table = table.copy()
    table["score_sound"] = scores
    table.to_csv(args.output_dir / "all_scores.csv", index=False)

    thresholds = np.round(np.arange(float(args.start), float(args.stop) + 1e-9, float(args.step)), 4)
    rows = []
    sweeps: dict[str, pd.DataFrame] = {}
    for split in ("val", "test"):
        split_table = table[table["split"].eq(split)].reset_index(drop=True)
        y_true = split_table["label"].astype(int).to_numpy()
        split_scores = split_table["score_sound"].astype(float).to_numpy()
        sweep = _sweep(y_true, split_scores, thresholds)
        sweep.insert(0, "split", split)
        sweep.to_csv(args.output_dir / f"{split}_threshold_sweep.csv", index=False)
        sweeps[split] = sweep

    picks = _pick_thresholds(sweeps["val"], float(args.clean_precision))
    for mode, threshold in picks.items():
        for split, sweep in sweeps.items():
            idx = (sweep["threshold"] - threshold).abs().idxmin()
            row = sweep.loc[idx].to_dict()
            row.update({"mode": mode, "selected_on": "val", "model": args.model_name})
            rows.append(row)
    selected = pd.DataFrame(rows)
    selected.to_csv(args.output_dir / "selected_thresholds.csv", index=False)
    _plot_sweep(sweeps["val"], picks, args.output_dir / "val_threshold_sweep.png", f"{args.model_name}: validation threshold sweep")

    summary = {
        "model": args.model_name,
        "clean_precision": float(args.clean_precision),
        "thresholds": picks,
        "selected_thresholds_csv": str(args.output_dir / "selected_thresholds.csv"),
    }
    (args.output_dir / "threshold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-precision", type=float, default=0.85)
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
