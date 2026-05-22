"""Summarize long-file inference and draw model overlay plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


MODELS = [
    ("perch_v2", "Perch 2.0", "#2A9D8F"),
    ("voxaboxen_beats", "Voxaboxen", "#E76F51"),
    ("animal2vec_pretrained_meerkat", "animal2vec", "#5E60CE"),
    ("wav2vec2_base", "Wav2Vec2", "#D9A441"),
]

PREDICTION_DIRS = {
    "perch_v2": "perch_v2",
    "voxaboxen_beats": "voxaboxen_beats",
    "animal2vec_pretrained_meerkat": "animal2vec_pretrained_meerkat",
    "wav2vec2_base": "wav2vec2_base",
}

BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#24343A"
GRID_COLOR = "#DCE6E8"
SOUND_COLOR = "#6EC6B8"


def _parse_models(value: str | None) -> list[tuple[str, str, str]]:
    if not value:
        return MODELS
    out: list[tuple[str, str, str]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split("|")
        if len(parts) != 3:
            raise ValueError("--models entries must be formatted as key|label|color")
        out.append((parts[0], parts[1], parts[2]))
    return out


def _load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        audio = signal.resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return audio, sr


def _load_predictions(predictions_dir: Path, models: list[tuple[str, str, str]]) -> dict[str, pd.DataFrame]:
    out = {}
    for key, _, _ in models:
        path = predictions_dir / PREDICTION_DIRS.get(key, key) / "window_predictions.csv"
        if path.exists():
            out[key] = pd.read_csv(path)
    return out


def _write_summary(predictions_dir: Path, out_dir: Path, models: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for key, label, _ in models:
        metrics_path = predictions_dir / PREDICTION_DIRS.get(key, key) / "inference_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": label,
                "windows": metrics.get("windows"),
                "predicted_sound_windows": metrics.get("predicted_sound_windows"),
                "predicted_sound_events": metrics.get("predicted_sound_events"),
                "precision": metrics.get("window_precision_sound"),
                "recall": metrics.get("window_recall_sound"),
                "f1": metrics.get("window_f1_sound"),
                "average_precision": metrics.get("window_average_precision"),
                "roc_auc": metrics.get("window_roc_auc"),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "inference_summary.csv", index=False)
    return summary


def _write_error_summary(predictions: dict[str, pd.DataFrame], out_dir: Path, models: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for key, label, _ in models:
        df = predictions.get(key)
        if df is None or "label" not in df:
            continue
        work = df.copy()
        work["segment_index"] = work["filename"].astype(str).str.split("_", n=1).str[0].astype(int)
        y = work["label"].astype(int)
        p = work["pred"].astype(int)
        work["tp"] = ((y == 1) & (p == 1)).astype(int)
        work["fp"] = ((y == 0) & (p == 1)).astype(int)
        work["fn"] = ((y == 1) & (p == 0)).astype(int)
        work["tn"] = ((y == 0) & (p == 0)).astype(int)
        for seg, seg_df in work.groupby("segment_index"):
            tp = int(seg_df["tp"].sum())
            fp = int(seg_df["fp"].sum())
            fn = int(seg_df["fn"].sum())
            tn = int(seg_df["tn"].sum())
            precision = tp / (tp + fp) if tp + fp else np.nan
            recall = tp / (tp + fn) if tp + fn else np.nan
            rows.append(
                {
                    "model": label,
                    "model_key": key,
                    "segment_index": int(seg),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "mean_score": float(seg_df["score_sound"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "error_summary_by_segment.csv", index=False)
    return out


def _plot_error_counts(error_summary: pd.DataFrame, out_dir: Path, models: list[tuple[str, str, str]]) -> None:
    if error_summary.empty:
        return
    total = error_summary.groupby("model", as_index=False)[["tp", "fp", "fn"]].sum()
    order = [label for _, label, _ in models if label in set(total["model"])]
    total["model"] = pd.Categorical(total["model"], categories=order, ordered=True)
    total = total.sort_values("model")

    colors = {"tp": "#2A9D8F", "fp": "#E76F51", "fn": "#5E60CE"}
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    bottom = np.zeros(len(total))
    x = np.arange(len(total))
    for col, label in [("tp", "true sound found"), ("fp", "false sound"), ("fn", "missed sound")]:
        values = total[col].astype(float).to_numpy()
        ax.bar(x, values, bottom=bottom, color=colors[col], alpha=0.88, label=label)
        bottom += values
    ax.set_xticks(x, total["model"].astype(str), rotation=0)
    ax.set_ylabel("windows")
    ax.set_title("Inference review: window error counts by model", color=TEXT_COLOR, pad=10)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_dir / "02_window_error_counts.png", dpi=170, facecolor=BG_COLOR)
    plt.close(fig)


def _plot_timeline(
    events: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    out_dir: Path,
    models: list[tuple[str, str, str]],
) -> None:
    duration_min = max(
        [float(events["global_end_s"].max()) / 60]
        + [float(df["global_end_s"].max()) / 60 for df in predictions.values() if "global_end_s" in df]
    )
    fig, axes = plt.subplots(
        len(models) + 1,
        1,
        figsize=(14, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [0.55] + [1] * len(models), "hspace": 0.1},
        facecolor=BG_COLOR,
    )
    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#AAB7BD")

    ax_ann = axes[0]
    for row in events.itertuples(index=False):
        ax_ann.axvspan(float(row.global_start_s) / 60, float(row.global_end_s) / 60, color=SOUND_COLOR, alpha=0.8, lw=0)
    ax_ann.set_ylim(0, 1)
    ax_ann.set_yticks([])
    ax_ann.set_ylabel("manual", rotation=0, ha="right", va="center", labelpad=45)
    ax_ann.set_title("Inference review: manual annotation and model scores", color=TEXT_COLOR, pad=10)

    for ax, (key, label, color) in zip(axes[1:], models, strict=False):
        df = predictions.get(key)
        ax.set_ylim(-0.08, 1.08)
        ax.axhline(0.5, color="#9BA8AE", lw=0.9, ls="--")
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=45)
        if df is None:
            ax.text(0.01, 0.5, "missing predictions", transform=ax.transAxes, color="#8A3A3A")
            continue
        x = (df["global_start_s"].astype(float) + df["global_end_s"].astype(float)) / 120
        ax.plot(x, df["score_sound"].astype(float), color=color, lw=1.2)
        positives = df.loc[df["pred"].astype(int).eq(1)]
        for row in positives.itertuples(index=False):
            ax.axvspan(float(row.global_start_s) / 60, float(row.global_end_s) / 60, color=color, alpha=0.12, lw=0)
        ax.text(0.995, 0.82, "score", transform=ax.transAxes, ha="right", color=color, fontsize=9)

    axes[-1].set_xlim(0, duration_min)
    axes[-1].set_xlabel("time, min")
    fig.tight_layout()
    fig.savefig(out_dir / "01_inference_scores_timeline.png", dpi=170, facecolor=BG_COLOR)
    plt.close(fig)


def _plot_segment_overlay(
    segment_index: int,
    base_dir: Path,
    events: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    out_dir: Path,
    target_sr: int,
    max_freq_hz: int,
    models: list[tuple[str, str, str]],
    output_name: str | None = None,
    title: str | None = None,
) -> None:
    order = pd.read_csv(base_dir / "label_studio_segments_ordered.csv")
    row = order.loc[order["index"].astype(int).eq(segment_index)].iloc[0]
    filename = str(row["file"])
    global_start = float(row["global_start_s"])
    global_end = float(row["global_end_s"])
    audio, sr = _load_audio(base_dir / "label_studio_segments_ordered" / filename, target_sr)
    duration = len(audio) / sr

    freqs, times, spec = signal.spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=2048,
        noverlap=1536,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    freq_mask = freqs <= min(max_freq_hz, sr // 2)
    spec_db = 20 * np.log10(spec[freq_mask] + 1e-8)
    vmin, vmax = np.percentile(spec_db, [5, 99.5])

    fig, (ax_spec, ax_score) = plt.subplots(
        2,
        1,
        figsize=(14, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1.45], "hspace": 0.08},
        facecolor=BG_COLOR,
    )
    for ax in (ax_spec, ax_score):
        ax.set_facecolor(BG_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AAB7BD")
        ax.spines["bottom"].set_color("#AAB7BD")

    im = ax_spec.pcolormesh(
        times,
        freqs[freq_mask] / 1000,
        spec_db,
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax_spec.set_ylabel("frequency, kHz")
    ax_spec.set_ylim(0, min(max_freq_hz, sr // 2) / 1000)
    ax_spec.set_title(title or f"Segment {segment_index:03d}: manual annotation and model scores", color=TEXT_COLOR, pad=10)

    seg_events = events[(events["global_end_s"] > global_start) & (events["global_start_s"] < global_end)]
    for ev in seg_events.itertuples(index=False):
        start = max(float(ev.global_start_s), global_start) - global_start
        end = min(float(ev.global_end_s), global_end) - global_start
        ax_spec.axvspan(start, end, color=SOUND_COLOR, alpha=0.28, lw=0)
        ax_score.axvspan(start, end, color=SOUND_COLOR, alpha=0.18, lw=0)

    for key, label, color in models:
        df = predictions.get(key)
        if df is None:
            continue
        seg = df[(df["global_end_s"] > global_start) & (df["global_start_s"] < global_end)].copy()
        if seg.empty:
            continue
        x = ((seg["global_start_s"].astype(float) + seg["global_end_s"].astype(float)) / 2) - global_start
        ax_score.plot(x, seg["score_sound"].astype(float), color=color, lw=1.4, label=label)

    ax_score.axhline(0.5, color="#9BA8AE", lw=0.9, ls="--")
    ax_score.set_ylim(-0.05, 1.05)
    ax_score.set_ylabel("score")
    ax_score.set_xlabel("time in segment, s")
    ax_score.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.colorbar(im, ax=ax_spec, pad=0.01, fraction=0.025).set_label("dB")
    fig.tight_layout()
    fig.savefig(
        out_dir / (output_name or f"segment_{segment_index:03d}_inference_overlay.png"),
        dpi=170,
        facecolor=BG_COLOR,
    )
    plt.close(fig)


def summarize(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = _parse_models(args.models)
    predictions = _load_predictions(args.predictions_dir, models)
    events = pd.read_csv(args.events)
    _write_summary(args.predictions_dir, args.output_dir, models)
    error_summary = _write_error_summary(predictions, args.output_dir, models)
    _plot_error_counts(error_summary, args.output_dir, models)
    _plot_timeline(events, predictions, args.output_dir, models)
    segments = [int(item) for item in str(args.segments).split(",") if str(item).strip()]
    if not segments:
        segments = [int(args.segment)]
    for segment in segments:
        _plot_segment_overlay(
            segment,
            args.base_dir,
            events,
            predictions,
            args.output_dir,
            args.target_sr,
            args.max_freq_hz,
            models,
        )
    if not error_summary.empty:
        perch_errors = error_summary.loc[error_summary["model_key"].eq("perch_v2")]
        if not perch_errors.empty:
            fp_seg = int(perch_errors.sort_values(["fp", "fn"], ascending=False).iloc[0]["segment_index"])
            fn_seg = int(perch_errors.sort_values(["fn", "fp"], ascending=False).iloc[0]["segment_index"])
            examples = [
                (fp_seg, "03_perch_false_positive_example.png", "Perch false-positive example"),
                (fn_seg, "04_perch_missed_sound_example.png", "Perch missed-sound example"),
            ]
            for seg, name, title_text in examples:
                _plot_segment_overlay(
                    seg,
                    args.base_dir,
                    events,
                    predictions,
                    args.output_dir,
                    args.target_sr,
                    args.max_freq_hz,
                    models,
                    output_name=name,
                    title=f"Segment {seg:03d}: {title_text}",
                )
    print(f"Wrote inference review summary to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("data/long_file_A/orcasound_2020-06-26-SRKW-Lpod"))
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/inference_review/orcasound_2020-06-26-SRKW-Lpod/predictions"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("data/long_file_A/orcasound_2020-06-26-SRKW-Lpod/annotations/annotation3_long_A_sound_events_merged.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/inference_review/orcasound_2020-06-26-SRKW-Lpod"),
    )
    parser.add_argument("--segment", type=int, default=13)
    parser.add_argument("--segments", default="")
    parser.add_argument("--target-sr", type=int, default=96000)
    parser.add_argument("--max-freq-hz", type=int, default=48000)
    parser.add_argument(
        "--models",
        default="",
        help="Semicolon-separated model config entries: key|label|color",
    )
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
