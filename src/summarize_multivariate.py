"""Create aggregate analysis tables from a multivariate RT-FM batch run.

By default this script reads the paper-style output directory produced by
``src/run_multivariate.py``. Change RESULTS_PATH/OUT_DIR below or copy this
script when analyzing another run directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_PATH = Path("outputs/multivariate_batch_clean_full/multivariate_batch_results.csv")
OUT_DIR = Path("outputs/multivariate_batch_clean_full/analysis")

MODEL_ORDER = [
    "uniform",
    "amplitude",
    "raw_radius_b1_w0.2",
    "raw_radius_b2_w0.2",
    "residual_tail_b1_w0.2",
    "residual_tail_b2_w0.2",
    "residual_tail_b4_w0.2",
]

MODEL_LABELS = {
    "uniform": "Uniform",
    "amplitude": "Amplitude",
    "raw_radius_b1_w0.2": "Raw b1",
    "raw_radius_b2_w0.2": "Raw b2",
    "residual_tail_b1_w0.2": "Residual b1",
    "residual_tail_b2_w0.2": "Residual b2",
    "residual_tail_b4_w0.2": "Residual b4",
}

COLORS = {
    "uniform": "#6B7280",
    "amplitude": "#D97706",
    "raw_radius_b1_w0.2": "#60A5FA",
    "raw_radius_b2_w0.2": "#2563EB",
    "residual_tail_b1_w0.2": "#86EFAC",
    "residual_tail_b2_w0.2": "#22C55E",
    "residual_tail_b4_w0.2": "#15803D",
}

PRIMARY_METRICS = [
    "ts_AUROC",
    "ts_AP_AUPRC",
    "ts_VUS_PR",
    "ts_best_F1",
    "ts_FP_rate@95R",
    "ts_tail_q95_FP_rate@95R",
    "event_precision@95R",
    "event_F1@95R",
]

DIAGNOSTIC_COLS = [
    "test_anomaly_rate",
    "test_window_anomaly_rate",
    "test_rare_q90_frac",
    "test_rare_q95_frac",
    "raw_radius_amp_corr",
    "resid_amp_corr",
    "vae_kl",
    "vae_rec",
    "train_windows",
    "test_windows",
]


def ordered_model_frame(frame):
    frame = frame.copy()
    frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
    return frame.sort_values("model")


def save_bar(summary, metric, ylabel, filename, lower_better=False):
    data = ordered_model_frame(summary[["model", metric]].dropna())
    labels = [MODEL_LABELS[str(m)] for m in data["model"]]
    values = data[metric].to_numpy()
    colors = [COLORS[str(m)] for m in data["model"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(np.arange(len(values)), values, color=colors, alpha=0.88)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    arrow = "↓" if lower_better else "↑"
    ax.set_ylabel(f"{ylabel} {arrow}")
    ax.set_title(f"Multivariate benchmark: {ylabel}")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=250, bbox_inches="tight")
    fig.savefig(OUT_DIR / filename.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_delta_plot(delta_summary):
    metrics = [
        ("ts_AP_AUPRC_delta", "AP delta"),
        ("ts_VUS_PR_delta", "VUS-PR delta"),
        ("ts_FP_rate@95R_delta", "FP@95R reduction"),
        ("event_F1@95R_delta", "Event F1 delta"),
    ]
    data = ordered_model_frame(delta_summary.reset_index())
    data = data[data["model"] != "uniform"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        values = data[metric].to_numpy()
        labels = [MODEL_LABELS[str(m)] for m in data["model"]]
        colors = [COLORS[str(m)] for m in data["model"]]
        ax.axhline(0, color="black", lw=0.8, alpha=0.55)
        ax.bar(np.arange(len(values)), values, color=colors, alpha=0.88)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Timestamp NLL improvements relative to Uniform FM", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "delta_vs_uniform_timestamp_nll.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUT_DIR / "delta_vs_uniform_timestamp_nll.pdf", bbox_inches="tight")
    plt.close(fig)


def save_dataset_heatmap(per_dataset, metric, filename, lower_better=False):
    pivot = per_dataset.pivot(index="dataset", columns="model", values=metric)
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    cmap = "viridis_r" if lower_better else "viridis"
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in pivot.columns], rotation=35, ha="right")
    ax.set_title(metric)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            ax.text(c, r, f"{values[r, c]:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=250, bbox_inches="tight")
    fig.savefig(OUT_DIR / filename.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_PATH)

    run_count = df[["dataset", "model", "seed"]].drop_duplicates().shape[0]
    if run_count != 84:
        raise RuntimeError(f"Expected 84 runs, found {run_count}")

    timestamp_nll = df[(df["level"] == "timestamp") & (df["score"] == "NLL")].copy()
    window_nll = df[(df["level"] == "window") & (df["score"] == "NLL")].copy()

    overall = (
        timestamp_nll.groupby("model")[PRIMARY_METRICS]
        .mean(numeric_only=True)
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    overall.to_csv(OUT_DIR / "overall_timestamp_nll_summary.csv", index=False)

    per_dataset = (
        timestamp_nll.groupby(["dataset", "model"])[PRIMARY_METRICS]
        .mean(numeric_only=True)
        .reset_index()
    )
    per_dataset["model"] = pd.Categorical(per_dataset["model"], MODEL_ORDER, ordered=True)
    per_dataset = per_dataset.sort_values(["dataset", "model"])
    per_dataset.to_csv(OUT_DIR / "per_dataset_timestamp_nll_summary.csv", index=False)

    score_family = (
        df[df["level"] == "timestamp"]
        .groupby(["score", "model"])[PRIMARY_METRICS]
        .mean(numeric_only=True)
        .reset_index()
    )
    score_family["model"] = pd.Categorical(score_family["model"], MODEL_ORDER, ordered=True)
    score_family.to_csv(OUT_DIR / "timestamp_score_family_summary.csv", index=False)

    diagnostics = (
        timestamp_nll.groupby("dataset")[DIAGNOSTIC_COLS]
        .mean(numeric_only=True)
        .reset_index()
    )
    diagnostics.to_csv(OUT_DIR / "dataset_diagnostics.csv", index=False)

    window_summary = (
        window_nll.groupby(["dataset", "model"])[
            [
                "win_AUROC",
                "win_AP_AUPRC",
                "win_best_F1",
                "win_FP_rate@95R",
                "win_tail_q95_FP_rate@95R",
                "hundman_window_precision@95R",
                "hundman_window_F1@95R",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    window_summary.to_csv(OUT_DIR / "per_dataset_window_nll_summary.csv", index=False)

    uniform = timestamp_nll[timestamp_nll["model"] == "uniform"][
        ["dataset", "seed"] + PRIMARY_METRICS
    ].rename(columns={metric: f"{metric}_uniform" for metric in PRIMARY_METRICS})
    deltas = timestamp_nll.merge(uniform, on=["dataset", "seed"], how="left")
    for metric in ["ts_AUROC", "ts_AP_AUPRC", "ts_VUS_PR", "ts_best_F1", "event_precision@95R", "event_F1@95R"]:
        deltas[f"{metric}_delta"] = deltas[metric] - deltas[f"{metric}_uniform"]
    for metric in ["ts_FP_rate@95R", "ts_tail_q95_FP_rate@95R"]:
        deltas[f"{metric}_delta"] = deltas[f"{metric}_uniform"] - deltas[metric]
    delta_cols = [col for col in deltas.columns if col.endswith("_delta")]
    delta_summary = deltas.groupby("model")[delta_cols].mean(numeric_only=True).reindex(MODEL_ORDER)
    delta_summary.to_csv(OUT_DIR / "delta_vs_uniform_timestamp_nll.csv")

    best_rows = []
    best_specs = {
        "best_AP": ("ts_AP_AUPRC", False),
        "best_VUS_PR": ("ts_VUS_PR", False),
        "best_FP95": ("ts_FP_rate@95R", True),
        "best_tail_FP95": ("ts_tail_q95_FP_rate@95R", True),
        "best_event_F1": ("event_F1@95R", False),
    }
    for label, (metric, lower_better) in best_specs.items():
        grouped = per_dataset[["dataset", "model", metric]].copy()
        idx = grouped.groupby("dataset")[metric].idxmin() if lower_better else grouped.groupby("dataset")[metric].idxmax()
        tmp = grouped.loc[idx].copy()
        tmp.insert(1, "criterion", label)
        tmp = tmp.rename(columns={metric: "value"})
        best_rows.append(tmp)
    best = pd.concat(best_rows, ignore_index=True).sort_values(["dataset", "criterion"])
    best.to_csv(OUT_DIR / "best_model_by_dataset_and_metric.csv", index=False)

    save_bar(overall, "ts_AP_AUPRC", "Timestamp AP/AUPRC", "overall_timestamp_ap.png")
    save_bar(overall, "ts_VUS_PR", "Timestamp VUS-PR", "overall_timestamp_vus_pr.png")
    save_bar(overall, "ts_FP_rate@95R", "Timestamp FP rate at 95% recall", "overall_timestamp_fp95.png", lower_better=True)
    save_bar(overall, "ts_tail_q95_FP_rate@95R", "Rare-normal FP rate at 95% recall", "overall_tail_fp95.png", lower_better=True)
    save_bar(overall, "event_F1@95R", "Event F1 at 95% recall", "overall_event_f1.png")

    save_delta_plot(delta_summary)
    save_dataset_heatmap(per_dataset, "ts_AP_AUPRC", "heatmap_dataset_timestamp_ap.png")
    save_dataset_heatmap(per_dataset, "ts_VUS_PR", "heatmap_dataset_timestamp_vus_pr.png")
    save_dataset_heatmap(per_dataset, "ts_FP_rate@95R", "heatmap_dataset_timestamp_fp95.png", lower_better=True)
    save_dataset_heatmap(per_dataset, "event_F1@95R", "heatmap_dataset_event_f1.png")

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "# Clean Multivariate Benchmark Summary\n\n"
        "Primary score: timestamp-level NLL.\n\n"
        "Key outputs:\n"
        "- `overall_timestamp_nll_summary.csv`\n"
        "- `per_dataset_timestamp_nll_summary.csv`\n"
        "- `delta_vs_uniform_timestamp_nll.csv`\n"
        "- `best_model_by_dataset_and_metric.csv`\n"
        "- `dataset_diagnostics.csv`\n"
        "- PNG/PDF plots for overall metrics, deltas, and per-dataset heatmaps.\n",
        encoding="utf-8",
    )

    print("Saved analysis to", OUT_DIR)
    print("\nOverall timestamp NLL:")
    print(overall.round(4).to_string(index=False))
    print("\nDeltas vs uniform, positive is better:")
    print(delta_summary.round(4).to_string())


if __name__ == "__main__":
    main()
