"""Plot long-file spectrogram checks with manual annotation overlays."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


SOUND_COLOR = "#2A9D8F"
WAVE_COLOR = "#6A8EAE"
TEXT_COLOR = "#24343A"
GRID_COLOR = "#DCE6E8"
BG_COLOR = "#FFFFFF"


def _read_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        audio = signal.resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return audio, sr


def _plot_timeline(order: pd.DataFrame, events: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 2.8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    sound_events = events.loc[events["label"].eq("sound")]
    for _, row in sound_events.iterrows():
        ax.axvspan(float(row.global_start_s) / 60, float(row.global_end_s) / 60, color=SOUND_COLOR, alpha=0.75, lw=0)

    for _, row in order.iterrows():
        ax.axvline(float(row.global_start_s) / 60, color="#EEF3F4", lw=0.8, zorder=0)

    ax.set_xlim(0, float(order["global_end_s"].max()) / 60)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time, min")
    ax.set_title("Inference review: manual annotation timeline", color=TEXT_COLOR, pad=10)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AAB7BD")
    ax.plot([], [], color=SOUND_COLOR, linewidth=8, label="sound annotation")
    ax.text(
        0.01,
        0.92,
        "unshaded background = noise",
        transform=ax.transAxes,
        color=TEXT_COLOR,
        fontsize=9,
        va="top",
    )
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "00_annotation_timeline.png", dpi=170, facecolor=BG_COLOR)
    plt.close(fig)


def _plot_segment(
    idx: int,
    file_name: str,
    path: Path,
    global_start: float,
    global_end: float,
    seg_events: pd.DataFrame,
    out_dir: Path,
    target_sr: int,
    max_freq_hz: int,
) -> dict:
    audio, sr = _read_audio(path, target_sr)
    duration = len(audio) / sr

    env_bins = min(1000, max(100, int(duration * 15)))
    edges = np.linspace(0, len(audio), env_bins + 1, dtype=int)
    env_t: list[float] = []
    env: list[float] = []
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        if b <= a:
            continue
        chunk = audio[a:b]
        env_t.append(((a + b) / 2) / sr)
        env.append(float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0)
    env_np = np.asarray(env)
    if env_np.size and env_np.max() > 0:
        env_np = env_np / env_np.max()

    nperseg = 2048
    noverlap = 1536
    freqs, times, spec = signal.spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    max_freq_hz = min(max_freq_hz, sr // 2)
    freq_mask = freqs <= max_freq_hz
    spec_db = 20 * np.log10(spec[freq_mask] + 1e-8)
    vmin, vmax = np.percentile(spec_db, [5, 99.5])

    fig, (ax_env, ax_spec) = plt.subplots(
        2,
        1,
        figsize=(13, 6.3),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.08},
        facecolor=BG_COLOR,
    )
    for ax in (ax_env, ax_spec):
        ax.set_facecolor(BG_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AAB7BD")
        ax.spines["bottom"].set_color("#AAB7BD")

    if len(env_t):
        ax_env.fill_between(env_t, 0, env_np, color=WAVE_COLOR, alpha=0.55, linewidth=0)
    ax_env.set_ylim(0, 1.05)
    ax_env.set_ylabel("level")
    ax_env.set_title(
        f"Segment {idx:03d}: {file_name}   global {global_start / 60:.2f}-{global_end / 60:.2f} min",
        color=TEXT_COLOR,
        pad=10,
    )
    ax_env.grid(axis="y", color=GRID_COLOR, linewidth=0.7)

    im = ax_spec.pcolormesh(
        times,
        freqs[freq_mask] / 1000,
        spec_db,
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax_spec.set_xlim(0, duration)
    ax_spec.set_ylim(0, max_freq_hz / 1000)
    ax_spec.set_ylabel("frequency, kHz")
    ax_spec.set_xlabel("time in segment, s")
    ax_spec.grid(color="#FFFFFF", alpha=0.08, linewidth=0.5)

    for _, ev in seg_events.iterrows():
        start = float(ev.plot_start_s)
        end = float(ev.plot_end_s)
        if ev.label == "sound":
            for ax in (ax_env, ax_spec):
                ax.axvspan(start, end, color=SOUND_COLOR, alpha=0.28, lw=0)

    ax_env.plot([], [], color=SOUND_COLOR, linewidth=8, label="sound annotation")
    ax_env.text(
        0.01,
        0.78,
        "unshaded background = noise",
        transform=ax_env.transAxes,
        color=TEXT_COLOR,
        fontsize=9,
        va="top",
    )
    ax_env.legend(frameon=False, loc="upper right")
    cbar = fig.colorbar(im, ax=ax_spec, pad=0.01, fraction=0.025)
    cbar.set_label("dB")
    fig.tight_layout()
    out_path = out_dir / f"segment_{idx:03d}_spectrogram.png"
    fig.savefig(out_path, dpi=170, facecolor=BG_COLOR)
    plt.close(fig)

    return {
        "index": idx,
        "file": file_name,
        "figure": out_path.name,
        "global_start_s": global_start,
        "global_end_s": global_end,
        "sound_events": int((seg_events["label"] == "sound").sum()),
    }


def plot_review(args: argparse.Namespace) -> None:
    base = args.base_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    order = pd.read_csv(base / "label_studio_segments_ordered.csv")
    events = pd.read_csv(base / "annotations" / "annotation3_long_A_sound_events_merged.csv")

    _plot_timeline(order, events, out_dir)

    rows = []
    segments_dir = base / "label_studio_segments_ordered"
    for _, seg in order.iterrows():
        idx = int(seg["index"])
        file_name = str(seg["file"])
        global_start = float(seg["global_start_s"])
        global_end = float(seg["global_end_s"])
        seg_events = events[
            (events["global_end_s"].astype(float) > global_start)
            & (events["global_start_s"].astype(float) < global_end)
        ].copy()
        seg_events["plot_start_s"] = seg_events["global_start_s"].astype(float).clip(global_start, global_end) - global_start
        seg_events["plot_end_s"] = seg_events["global_end_s"].astype(float).clip(global_start, global_end) - global_start
        rows.append(
            _plot_segment(
                idx,
                file_name,
                segments_dir / file_name,
                global_start,
                global_end,
                seg_events,
                out_dir,
                args.target_sr,
                args.max_freq_hz,
            )
        )

    with (out_dir / "spectrogram_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "README.txt").write_text(
        "\n".join(
            [
                "Inference review spectrograms.",
                "",
                "Open first:",
                "00_annotation_timeline.png",
                "",
                "Per-segment files:",
                "segment_001_spectrogram.png ... segment_030_spectrogram.png",
                "",
                "Overlay colors:",
                "green/blue: manual sound annotation",
                "unshaded background: noise",
                "",
                "Index:",
                "spectrogram_index.csv",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} segment spectrograms to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/long_file_A/orcasound_2020-06-26-SRKW-Lpod"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/inference_review/orcasound_2020-06-26-SRKW-Lpod"),
    )
    parser.add_argument("--target-sr", type=int, default=96000)
    parser.add_argument("--max-freq-hz", type=int, default=48000)
    return parser.parse_args()


if __name__ == "__main__":
    plot_review(parse_args())
