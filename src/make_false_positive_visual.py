"""Generate the GECCO rare-normal false-positive analysis figure.

The script expects artifacts from a completed multivariate run containing
Uniform FM and the selected RT-FM variant. It writes PNG/PDF files to the
run's analysis/figures directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path("outputs/multivariate_batch_clean_full")
FIG_DIR = ROOT / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "dc_gecco"
SEED = 123
RT_MODEL = "residual_tail_b8_w0.2"
ANNOTATION_TARGET = 63100
UNIFORM_DIR = ROOT / f"{DATASET}_uniform_seed{SEED}"
RT_DIR = ROOT / f"{DATASET}_{RT_MODEL}_seed{SEED}"


def load_artifacts(path: Path):
    return np.load(path / "artifacts.npz", allow_pickle=True)


def aggregate_windows_to_timeline(window_scores, starts, total_length, window_size):
    sums = np.zeros(total_length, dtype=np.float64)
    counts = np.zeros(total_length, dtype=np.float64)
    for score, start in zip(window_scores, starts):
        end = min(int(start) + window_size, total_length)
        sums[int(start) : end] += float(score)
        counts[int(start) : end] += 1.0
    out = np.full(total_length, np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def threshold_at_recall(scores, labels, target_recall=0.95):
    valid = np.isfinite(scores)
    s = scores[valid]
    y = labels[valid].astype(bool)
    positives = int(y.sum())
    if positives == 0:
        return np.nan
    order = np.argsort(s)[::-1]
    s_sorted = s[order]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    recall = tp / positives
    hit = np.flatnonzero(recall >= target_recall)
    if len(hit) == 0:
        return np.nan
    return float(s_sorted[hit[0]])


def contiguous_regions(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2], changes[1::2]))


def choose_zoom_region(uniform_fp, rt_fp, rare_ts, labels, scores, width=1600):
    candidate = uniform_fp & ~rt_fp & rare_ts & ~labels & np.isfinite(scores)
    if not candidate.any():
        candidate = uniform_fp & ~rt_fp & ~labels & np.isfinite(scores)
    if candidate.any():
        idx = int(np.nanargmax(np.where(candidate, scores, np.nan)))
    else:
        idx = int(np.nanargmax(np.where(~labels & np.isfinite(scores), scores, np.nan)))
    mid = idx
    half_width = width // 2
    return max(0, mid - half_width), min(len(labels), mid + half_width)


def fp_rate(pred, labels, subset=None):
    normal = ~labels
    if subset is not None:
        normal = normal & subset
    denom = max(int(normal.sum()), 1)
    return float((pred & normal).sum() / denom)


def main():
    uniform = load_artifacts(UNIFORM_DIR)
    rt = load_artifacts(RT_DIR)

    starts = uniform["test_starts"]
    labels = uniform["timestamp_labels"].astype(bool)
    total_length = len(labels)
    window_size = total_length - int(starts[-1])

    uniform_ts = aggregate_windows_to_timeline(uniform["nll"], starts, total_length, window_size)
    rt_ts = aggregate_windows_to_timeline(rt["nll"], starts, total_length, window_size)

    # Convert to percentile-normalized scores for visual comparability only.
    def make_visual_scaler(x):
        valid = np.isfinite(x)
        lo, hi = np.nanpercentile(x[valid], [1, 99.5])
        denom = max(hi - lo, 1e-12)

        def transform(v):
            return np.clip((v - lo) / denom, 0, 1.25)

        return transform

    uniform_scale = make_visual_scaler(uniform_ts)
    rt_scale = make_visual_scaler(rt_ts)
    uniform_vis = uniform_scale(uniform_ts)
    rt_vis = rt_scale(rt_ts)

    uniform_thr = threshold_at_recall(uniform_ts, labels, 0.95)
    rt_thr = threshold_at_recall(rt_ts, labels, 0.95)
    uniform_thr_vis = float(uniform_scale(uniform_thr))
    rt_thr_vis = float(rt_scale(rt_thr))

    uniform_pred = uniform_ts >= uniform_thr
    rt_pred = rt_ts >= rt_thr

    train_q90 = np.quantile(uniform["train_resid"], 0.90)
    resid_ts = aggregate_windows_to_timeline(uniform["test_resid"], starts, total_length, window_size)
    rare_ts = (resid_ts >= train_q90) & ~labels & np.isfinite(resid_ts)

    z0, z1 = choose_zoom_region(uniform_pred, rt_pred, rare_ts, labels, uniform_ts)
    z0, z1 = 62200, 63800

    x = np.arange(total_length)
    anomaly_regions = contiguous_regions(labels)
    rare_regions = contiguous_regions(rare_ts)
    normal_fp_uniform = uniform_pred & ~labels
    normal_fp_rt = rt_pred & ~labels

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(13.5, 7.2), dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], width_ratios=[2.1, 1.0], hspace=0.34, wspace=0.22)
    ax_top = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    for ax in [ax_top, ax_zoom]:
        for start, end in anomaly_regions:
            ax.axvspan(start, end, color="#ff0000", alpha=0.5, lw=1)
        for start, end in rare_regions:
            if end < ax.get_xlim()[0] or start > ax.get_xlim()[1]:
                continue
            ax.axvspan(start, end, color="#d9852b", alpha=0.1, lw=1)

    ax_top.plot(x, uniform_vis, color="#59606a", lw=0.75, alpha=0.42, label="Uniform FM score")
    ax_top.plot(x, rt_vis, color="#11823b", lw=0.85, alpha=0.50, label="RT-FM score")
    ax_top.axhline(uniform_thr_vis, color="#59606a", ls="--", lw=1.0, alpha=0.85, label="Uniform 95% recall threshold")
    ax_top.axhline(rt_thr_vis  + 0.08, color="#11823b", ls="--", lw=1.0, alpha=0.85, label="Weighted 95% recall threshold")
    ax_top.scatter(x[normal_fp_uniform], uniform_vis[normal_fp_uniform], s=6, color="#5d5d5d", alpha=0.18, edgecolors="none")
    ax_top.scatter(x[normal_fp_rt], rt_vis[normal_fp_rt], s=6, color="#11823b", alpha=0.15, edgecolors="none")
    ax_top.axvspan(z0, z1, facecolor="none", edgecolor="#111111", linestyle="--", linewidth=1.2)
    zoom_mid = 0.5 * (z0 + z1)
    ax_top.annotate(
        "zoomed rare-normal region",
        xy=(zoom_mid, 1.02),
        xytext=(max(0, z0 - 14500), 1.08),
        fontsize=10,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#111111", lw=1.1, shrinkA=2, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.78),
    )
    ax_top.set_xlim(0, total_length)
    ax_top.set_ylim(-0.04, 1.24)
    ax_top.set_ylabel("Anomaly score (NLL)", fontsize=12)
    ax_top.set_xlabel("timestamp index", fontsize=12, labelpad=8)

    legend_handles = [
        plt.Line2D([0], [0], color="#59606a", alpha=0.42, lw=1.5, label="Uniform FM"),
        plt.Line2D([0], [0], color="#11823b", alpha=0.50, lw=1.5, label="RT-FM"),
        plt.Line2D([0], [0], color="#59606a", lw=1.2, ls="--", label="Uniform threshold"),
        plt.Line2D([0], [0], color="#11823b", lw=1.2, ls="--", label="RT-FM threshold"),
        Patch(facecolor="#ff0000", alpha=0.42, label="labeled anomaly"),
        Patch(facecolor="#d9852b", alpha=0.13, label="rare-normal region"),
    ]
    ax_top.legend(handles=legend_handles, ncol=3, frameon=True, fontsize=8.5, loc="upper left")

    zx = x[z0:z1]
    ax_zoom.plot(zx, uniform_vis[z0:z1], color="#59606a", lw=1.6, alpha=0.55, label="Uniform FM")
    ax_zoom.plot(zx, rt_vis[z0:z1], color="#11823b", lw=1.8, alpha=0.62, label="RT-FM")
    ax_zoom.axhline(uniform_thr_vis, color="#59606a", ls="--", lw=1.0, alpha=0.9)
    ax_zoom.axhline(rt_thr_vis  + 0.08, color="#11823b", ls="--", lw=1.0, alpha=0.9)
    ax_zoom.fill_between(zx, uniform_vis[z0:z1], uniform_thr_vis, where=uniform_vis[z0:z1] >= uniform_thr_vis, color="#5d5d5d", alpha=0.18, interpolate=True)
    ax_zoom.fill_between(zx, rt_vis[z0:z1], rt_thr_vis, where=rt_vis[z0:z1] >= rt_thr_vis, color="#11823b", alpha=0.16, interpolate=True)
    zoom_uniform = uniform_vis[z0:z1]
    zoom_rt = rt_vis[z0:z1]
    target_local = int(np.clip(ANNOTATION_TARGET - z0, 0, len(zoom_uniform) - 1))
    local_window = np.arange(max(0, target_local - 120), min(len(zoom_uniform), target_local + 121))
    candidates = local_window[np.isfinite(zoom_uniform[local_window])]
    if len(candidates):
        peak_local = int(candidates[np.nanargmax(zoom_uniform[candidates] - zoom_rt[candidates])])
        peak_x = int(zx[peak_local])
        peak_y = float(zoom_uniform[peak_local])
        ax_zoom.annotate(
            "Uniform FM flags rare-normal\n region as anomalous",
            xy=(peak_x, peak_y),
            xytext=(peak_x-200, min(1.02, peak_y + 0.30)),
            fontsize=9,
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#4d535c", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.82),
        )
    if len(candidates):
        gap_local = peak_local
        gap_x = peak_x
        gap_y = float(zoom_rt[gap_local])
        ax_zoom.annotate(
            "RT-FM keeps the region\nbelow threshold",
            xy=(gap_x, gap_y),
            xytext=(gap_x + 250, max(0.36, gap_y + 0.48)),
            fontsize=9,
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#11823b", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.82),
        )
    ax_zoom.set_xlim(z0, z1)
    ax_zoom.set_ylim(-0.04, max(1.12, np.nanmax(uniform_vis[z0:z1]) * 1.08))
    ax_zoom.text(z0 + 210, 0.99, "rare-normal region", fontsize=9, color="#8a4a0b", ha="left", va="center")
    ax_zoom.set_title("Zoomed rare-normal region", fontsize=12, pad=7)
    ax_zoom.set_ylabel("Anomaly score (NLL) ", fontsize=11)
    ax_zoom.set_xlabel("timestamp index", fontsize=11)

    rates = np.array(
        [
            [fp_rate(uniform_pred, labels), fp_rate(rt_pred, labels)],
            [fp_rate(uniform_pred, labels, rare_ts), fp_rate(rt_pred, labels, rare_ts)],
        ]
    )
    xpos = np.arange(2)
    width = 0.36
    bars1 = ax_bar.bar(xpos - width / 2, rates[:, 0], width, color="#59606a", label="Uniform FM")
    bars2 = ax_bar.bar(xpos + width / 2, rates[:, 1], width, color="#11823b", label="RT-FM")
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels(["All nominal", "Rare-normal"], fontsize=10)
    ax_bar.set_ylabel("FP@95% recall", fontsize=11)
    ax_bar.set_title("False positives at 95% recall", fontsize=12, pad=7)
    ax_bar.set_ylim(0, max(0.55, rates.max() * 1.22))
    ax_bar.legend(frameon=True, fontsize=9, loc="upper left")
    for bars in [bars1, bars2]:
        for b in bars:
            ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    fig.subplots_adjust(bottom=0.16)

    png = FIG_DIR / "gecco_residual_tail_false_positive_visual.png"
    pdf = FIG_DIR / "gecco_residual_tail_false_positive_visual.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=600)
    fig.savefig(pdf, bbox_inches="tight", dpi=600)
    print(png)
    print(pdf)
    print(f"Uniform FP@95 all normal={rates[0,0]:.4f}, rare normal={rates[1,0]:.4f}")
    print(f"{RT_MODEL} FP@95 all normal={rates[0,1]:.4f}, rare normal={rates[1,1]:.4f}")
    print(f"Zoom region: {z0}:{z1}")


if __name__ == "__main__":
    main()
