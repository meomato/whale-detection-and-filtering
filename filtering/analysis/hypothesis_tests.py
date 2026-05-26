"""Recompute the hypothesis-test numbers reported in docs/HYPOTHESIS.md."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MODEL_ORDER = ["Perch", "Voxaboxen", "animal2vec", "Wav2Vec2"]


def _bootstrap_mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    *,
    max_exact_resamples: int = 250_000,
    random_resamples: int = 200_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for the mean.

    Small cases are exhaustive, while larger cases use a seeded Monte Carlo
    bootstrap so the script stays fast and reproducible.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    exact_resamples = n**n
    if exact_resamples <= max_exact_resamples:
        means = np.fromiter(
            (values[list(sample)].mean() for sample in itertools.product(range(n), repeat=n)),
            dtype=float,
        )
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n, size=(random_resamples, n))
        means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def _one_sided_sign_permutation_p(values: np.ndarray) -> float:
    """Exact sign-flip p-value for H1: mean(values) > 0."""
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        means.append(float(np.mean(values * np.asarray(signs))))
    return float((sum(mean >= observed - 1e-12 for mean in means) + 1) / (len(means) + 1))


def _mcnemar_from_predictions(root: Path) -> tuple[int, int, float, float, int]:
    pred_root = root / "outputs/inference_review_context5_hop1/orcasound_2020-06-26-SRKW-Lpod/predictions_cv_annotations_all"
    perch = pd.read_csv(pred_root / "perch_v2/window_predictions.csv")
    voxaboxen = pd.read_csv(pred_root / "voxaboxen_beats/window_predictions.csv")
    merged = perch.merge(
        voxaboxen,
        on=["filename", "start_s", "end_s", "label"],
        suffixes=("_perch", "_voxaboxen"),
    )
    correct_perch = merged["pred_perch"].astype(int).eq(merged["label"].astype(int))
    correct_voxaboxen = merged["pred_voxaboxen"].astype(int).eq(merged["label"].astype(int))
    b = int((correct_perch & ~correct_voxaboxen).sum())
    c = int((~correct_perch & correct_voxaboxen).sum())
    chi2 = float((abs(b - c) - 1) ** 2 / (b + c))
    p_value = float(stats.chi2.sf(chi2, 1))
    return b, c, chi2, p_value, int(len(merged))


def run(root: Path) -> dict[str, object]:
    cv = pd.read_csv(root / "reports/benchmark_context5_hop1_cv/cv_fold_metrics.csv")
    cv_all = cv[cv["scenario"].eq("annotations_all")].copy()

    ap_by_fold = cv_all.pivot(index="fold", columns="model", values="average_precision")
    h1_diff = (
        ap_by_fold[["perch_v2", "voxaboxen_beats", "animal2vec_pretrained_meerkat"]].mean(axis=1)
        - ap_by_fold["wav2vec2_base"]
    ).to_numpy()
    h1_t = stats.ttest_1samp(h1_diff, 0.0, alternative="greater")

    fpr_by_fold = cv_all.pivot(index="fold", columns="model", values="false_positive_rate")
    h2_diff = fpr_by_fold[["perch_v2", "voxaboxen_beats"]].mean(axis=1) - fpr_by_fold["wav2vec2_base"]
    h2_t = stats.ttest_1samp(h2_diff.to_numpy(), 0.0, alternative="less")

    h3_event_ap = stats.spearmanr(cv_all["average_precision"], cv_all["voxaboxen_style_event_ap_0.5"])
    h3_event_f1 = stats.spearmanr(cv_all["average_precision"], cv_all["event_f1"])

    heads = pd.read_csv(root / "reports/tuning/tables/hpc_head_cv_fold_metrics.csv")
    heads["head_kind"] = np.where(heads["head"].str.contains("MLP2"), "mlp2", "linear")
    head_pairs = heads.pivot_table(index=["model", "fold"], columns="head_kind", values="event_f1")
    h4_diff = (head_pairs["mlp2"] - head_pairs["linear"]).to_numpy()
    h4_w = stats.wilcoxon(h4_diff, alternative="greater")

    inference = pd.read_csv(
        root / "reports/inference_review_context5_hop1/orcasound_2020-06-26-SRKW-Lpod_cv_annotations_all/inference_summary.csv"
    )
    benchmark_ap = pd.Series(
        {
            "Perch": float(cv_all.loc[cv_all["model"].eq("perch_v2"), "average_precision"].mean()),
            "Voxaboxen": float(cv_all.loc[cv_all["model"].eq("voxaboxen_beats"), "average_precision"].mean()),
            "animal2vec": float(
                cv_all.loc[cv_all["model"].eq("animal2vec_pretrained_meerkat"), "average_precision"].mean()
            ),
            "Wav2Vec2": float(cv_all.loc[cv_all["model"].eq("wav2vec2_base"), "average_precision"].mean()),
        }
    )
    long_file_ap = inference.set_index("model")["average_precision"].rename({"Perch 2.0": "Perch"})
    long_file_ap = long_file_ap.reindex(MODEL_ORDER)
    h5_spearman = stats.spearmanr(benchmark_ap.reindex(MODEL_ORDER), long_file_ap)
    b, c, chi2, mcnemar_p, shared_windows = _mcnemar_from_predictions(root)

    return {
        "H1": {
            "diff": h1_diff,
            "mean": float(h1_diff.mean()),
            "bootstrap_ci": _bootstrap_mean_ci(h1_diff),
            "t": float(h1_t.statistic),
            "p_one_sided_t": float(h1_t.pvalue),
            "p_one_sided_permutation": _one_sided_sign_permutation_p(h1_diff),
        },
        "H2": {
            "diff": h2_diff.to_numpy(),
            "mean": float(h2_diff.mean()),
            "t": float(h2_t.statistic),
            "p_one_sided_t": float(h2_t.pvalue),
        },
        "H3": {
            "window_ap_vs_event_ap_rho": float(h3_event_ap.statistic),
            "window_ap_vs_event_ap_p": float(h3_event_ap.pvalue),
            "window_ap_vs_event_f1_rho": float(h3_event_f1.statistic),
            "window_ap_vs_event_f1_p": float(h3_event_f1.pvalue),
        },
        "H4": {
            "n": int(len(h4_diff)),
            "mean": float(h4_diff.mean()),
            "median": float(np.median(h4_diff)),
            "bootstrap_ci": _bootstrap_mean_ci(h4_diff),
            "wilcoxon_w": float(h4_w.statistic),
            "p_one_sided": float(h4_w.pvalue),
        },
        "H5": {
            "benchmark_ap": benchmark_ap.reindex(MODEL_ORDER),
            "long_file_ap": long_file_ap,
            "spearman_rho": float(h5_spearman.statistic),
            "spearman_p": float(h5_spearman.pvalue),
            "shared_windows": shared_windows,
            "mcnemar_b": b,
            "mcnemar_c": c,
            "mcnemar_chi2": chi2,
            "mcnemar_p": mcnemar_p,
        },
    }


def _fmt(values: np.ndarray, digits: int = 3) -> str:
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def print_report(results: dict[str, object]) -> None:
    h1 = results["H1"]
    h2 = results["H2"]
    h3 = results["H3"]
    h4 = results["H4"]
    h5 = results["H5"]

    print("Hypothesis test recomputation")
    print()
    print("H1 specialized encoders outperform Wav2Vec2 in Window AP")
    print(f"  d = {_fmt(h1['diff'])}")
    print(f"  mean(d) = {h1['mean']:.3f}")
    print(f"  95% bootstrap CI = [{h1['bootstrap_ci'][0]:.3f}, {h1['bootstrap_ci'][1]:.3f}]")
    print(f"  t = {h1['t']:.3f}")
    print(f"  p_one_sided_t = {h1['p_one_sided_t']:.4f}")
    print(f"  p_one_sided_permutation = {h1['p_one_sided_permutation']:.4f}")
    print()
    print("H2 specialized encoders reduce false positives")
    print(f"  d = {_fmt(h2['diff'])}")
    print(f"  mean(d) = {h2['mean']:.3f}")
    print(f"  t = {h2['t']:.3f}")
    print(f"  p_one_sided_t = {h2['p_one_sided_t']:.4f}")
    print()
    print("H3 Window AP is not enough for event detection")
    print(f"  Window AP vs Event AP@0.5: rho = {h3['window_ap_vs_event_ap_rho']:.3f}, p = {h3['window_ap_vs_event_ap_p']:.3f}")
    print(f"  Window AP vs Event F1: rho = {h3['window_ap_vs_event_f1_rho']:.3f}, p = {h3['window_ap_vs_event_f1_p']:.4f}")
    print()
    print("H4 MLP2 heads improve Event F1 over linear heads")
    print(f"  n = {h4['n']}")
    print(f"  mean(d) = {h4['mean']:.4f}")
    print(f"  median(d) = {h4['median']:.4f}")
    print(f"  95% bootstrap CI = [{h4['bootstrap_ci'][0]:.4f}, {h4['bootstrap_ci'][1]:.4f}]")
    print(f"  Wilcoxon W = {h4['wilcoxon_w']:.0f}")
    print(f"  p_one_sided = {h4['p_one_sided']:.3f}")
    print()
    print("H5 benchmark ranking transfers to long-file review")
    print("  benchmark AP: " + ", ".join(f"{model} {value:.3f}" for model, value in h5["benchmark_ap"].items()))
    print("  long-file AP: " + ", ".join(f"{model} {value:.3f}" for model, value in h5["long_file_ap"].items()))
    print(f"  Spearman rho = {h5['spearman_rho']:.3f}, p = {h5['spearman_p']:.3f}")
    print(f"  shared windows = {h5['shared_windows']}")
    print(f"  b = {h5['mcnemar_b']}, c = {h5['mcnemar_c']}")
    print(f"  chi2 = {h5['mcnemar_chi2']:.3f}")
    print(f"  p = {h5['mcnemar_p']:.2e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_report(run(args.root))


if __name__ == "__main__":
    main()
