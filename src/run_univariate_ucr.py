"""Univariate UCR benchmark runner for RT-FM.

This script trains a temporal VAE for residual-tail discovery and then trains
Uniform FM, amplitude-weighted FM, raw-radius FM, and RT-FM variants on UCR
time-series anomaly-detection files. It is written as a self-contained runner
for anonymous review; dataset paths are supplied by command-line arguments and
all outputs are CSV/NPZ artifacts.
"""

import argparse
import itertools
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchcfm import TargetConditionalFlowMatcher

try:
    from vus.metrics import get_metrics as get_vus_metrics
except Exception:
    get_vus_metrics = None


DATASETS = [
    "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620",
    "011_UCR_Anomaly_DISTORTEDECG1_10000_11800_12100",
    "045_UCR_Anomaly_DISTORTEDPowerDemand2_14000_23357_23717",
    "113_UCR_Anomaly_CIMIS44AirTemperature1_4000_5391_5392",
    "123_UCR_Anomaly_ECG4_5000_16800_17100",
]
SEEDS = [42, 123]
# Default location expected by this anonymous release. Override with --data-dir.
DATA_DIR = Path("data/UCR_Anomaly_FullData")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def valid_group_count(channels, max_groups=8):
    for group_count in range(min(max_groups, channels), 0, -1):
        if channels % group_count == 0:
            return group_count
    return 1


class ResidualBlock1dWithGroupNorm(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        group_count = valid_group_count(channels, num_groups)
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
            nn.GroupNorm(group_count, channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
            nn.GroupNorm(group_count, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class TemporalVAE(nn.Module):
    def __init__(self, seq_len, latent_dim=24, hidden_size=64, num_blocks=3):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=3, padding="same"),
            nn.GroupNorm(valid_group_count(hidden_size), hidden_size),
            nn.SiLU(inplace=True),
            *[ResidualBlock1dWithGroupNorm(hidden_size) for _ in range(num_blocks)],
            nn.Flatten(),
        )
        self.to_mu = nn.Linear(hidden_size * seq_len, latent_dim)
        self.to_logvar = nn.Linear(hidden_size * seq_len, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_size * seq_len)
        self.decoder_blocks = nn.Sequential(
            *[ResidualBlock1dWithGroupNorm(hidden_size) for _ in range(num_blocks)],
            nn.Conv1d(hidden_size, 1, kernel_size=3, padding="same"),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def encode(self, x):
        h = self.encoder(x[:, None, :])
        return self.to_mu(h), self.to_logvar(h).clamp(-10.0, 10.0)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        h = self.decoder_input(z).view(z.shape[0], -1, self.seq_len)
        return self.decoder_blocks(h).squeeze(1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def vae_terms(x, x_hat, mu, logvar):
    rec = ((x_hat - x) ** 2).mean(dim=1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return rec, kl


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FMResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        # Match architecture_tuning_detection_demo.ipynb: keep GroupNorm in
        # the VAE, but use the older plain residual block in the FM vector field.
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TemporalFM(nn.Module):
    def __init__(self, seq_len, hidden_size=128, num_layers=5, time_embed_dim=None, num_groups=8):
        super().__init__()
        self.seq_len = seq_len
        self.time_embed_dim = time_embed_dim or hidden_size
        self.raw_time_dim = min(64, self.time_embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.raw_time_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.input_conv = nn.Conv1d(1, hidden_size, kernel_size=3, padding="same")
        self.cond_proj = nn.Linear(self.time_embed_dim + 2, hidden_size)
        self.res_blocks = nn.Sequential(*[FMResidualBlock(hidden_size, num_groups=num_groups) for _ in range(num_layers)])
        self.output_conv = nn.Conv1d(hidden_size, 1, kernel_size=3, padding="same")
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, timesteps, x):
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])
        time_emb = self.time_embed(timestep_embedding(timesteps.to(x.device), self.raw_time_dim))
        feature_vec = torch.stack([x.mean(dim=1), x.std(dim=1)], dim=1)
        cond = self.cond_proj(torch.cat([time_emb, feature_vec], dim=1))[:, :, None]
        h = self.input_conv(x[:, None, :]) + cond
        h = self.res_blocks(h)
        return self.output_conv(h).squeeze(1)


def make_windows(series, starts, window_size):
    return np.stack([series[s:s + window_size] for s in starts]).astype(np.float32)


def load_dataset(file_name, window_size=64, use_clipping=True, clip_value=10.0, scale_floor=1e-3, data_dir=DATA_DIR):
    parts = file_name.split("_")
    train_split, anomaly_start, anomaly_end = map(int, parts[-3:])
    values = np.loadtxt(Path(data_dir) / f"{file_name}.txt").astype(np.float32)
    center = np.nanmean(values[:train_split])
    scale = max(float(np.nanstd(values[:train_split])), scale_floor)
    scaled = np.nan_to_num((values - center) / scale).astype(np.float32)
    if use_clipping:
        scaled = np.clip(scaled, -clip_value, clip_value).astype(np.float32)
    train_starts = np.arange(0, train_split - window_size + 1)
    test_starts = np.arange(train_split, len(scaled) - window_size + 1)
    train_seqs = make_windows(scaled, train_starts, window_size)
    test_seqs = make_windows(scaled, test_starts, window_size)
    test_ends = test_starts + window_size - 1
    test_labels = (test_starts <= anomaly_end) & (test_ends >= anomaly_start)
    timestamp_labels = np.zeros(len(scaled), dtype=bool)
    timestamp_labels[anomaly_start:anomaly_end + 1] = True
    return train_seqs, test_seqs, test_labels, test_starts, timestamp_labels, train_split, anomaly_start, anomaly_end, len(scaled)


def train_vae(train_seqs, device, seed, latent_dim=24, iterations=30000, log_every=5000):
    dataset = TensorDataset(torch.tensor(train_seqs, dtype=torch.float, device=device))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=128, sampler=RandomSampler(dataset, generator=gen))
    iterator = itertools.cycle(loader)
    vae = TemporalVAE(train_seqs.shape[1], latent_dim=latent_dim).to(device)
    opt = optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-4)
    vae.train()
    for step in range(1, iterations + 1):
        (x,) = next(iterator)
        opt.zero_grad(set_to_none=True)
        x_hat, mu, logvar = vae(x)
        rec, kl = vae_terms(x, x_hat, mu, logvar)
        loss = rec.mean() + 0.05 * kl.mean()
        loss.backward()
        opt.step()
        if step == 1 or step % log_every == 0 or step == iterations:
            print(
                f"  VAE {step:05d}/{iterations} loss={float(loss.detach().cpu()):.4f} "
                f"rec={float(rec.mean().detach().cpu()):.4f} kl={float(kl.mean().detach().cpu()):.4f}",
                flush=True,
            )
    return vae


@torch.no_grad()
def encode_arrays(vae, seqs, device):
    loader = DataLoader(TensorDataset(torch.tensor(seqs, dtype=torch.float, device=device)), batch_size=2048)
    mus, logvars = [], []
    vae.eval()
    for (x,) in loader:
        mu, logvar = vae.encode(x)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
    mu = torch.cat(mus).numpy()
    logvar = torch.cat(logvars).numpy()
    return np.sum(mu ** 2 + np.exp(logvar), axis=1)


def bounded_weights(scores, boost=4.0):
    lo, hi = np.quantile(scores, [0.90, 0.99])
    strength = np.clip((scores - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    w = 1.0 + boost * strength
    return w / w.mean()


def best_f1_score(labels, scores):
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores)
    if labels.sum() == 0:
        return np.nan
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(~sorted_labels)
    fn = labels.sum() - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    # Predictions only change after the last occurrence of a tied score.
    last_for_score = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    return float(f1[last_for_score].max(initial=0.0))


def fp_at_recall(labels, scores, target_recall, subset_mask=None):
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores)
    if labels.sum() == 0:
        return {"threshold": np.nan, "achieved_recall": np.nan, "fp_rate": np.nan}
    if subset_mask is None:
        subset_mask = ~labels
    else:
        subset_mask = np.asarray(subset_mask).astype(bool)
    positives = scores[labels]
    thresholds = np.unique(positives)
    valid_thresholds = []
    for threshold in thresholds:
        recall = np.mean(positives >= threshold)
        if recall >= target_recall:
            valid_thresholds.append(threshold)
    if not valid_thresholds:
        return {"threshold": np.nan, "achieved_recall": np.nan, "fp_rate": np.nan}
    threshold = max(valid_thresholds)  # highest threshold still reaching recall
    denom = subset_mask.sum()
    return {
        "threshold": float(threshold),
        "achieved_recall": float(np.mean(positives >= threshold)),
        "fp_rate": float(np.sum((scores >= threshold) & subset_mask) / denom) if denom else np.nan,
    }


def detection_metrics(labels, scores):
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return {"auroc": np.nan, "auprc": np.nan, "best_f1": np.nan}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "best_f1": best_f1_score(labels, scores),
    }


def ranges(mask):
    mask = np.asarray(mask).astype(bool)
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return list(zip(starts.tolist(), ends.tolist()))


def merge_intervals(intervals):
    intervals = sorted((int(a), int(b)) for a, b in intervals if b >= a)
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def interval_overlap_metrics(gt_intervals, pred_intervals):
    tp = fp = 0
    hit = set()
    for ps, pe in pred_intervals:
        overlaps = [i for i, (gs, ge) in enumerate(gt_intervals) if not (pe < gs or ps > ge)]
        if overlaps:
            tp += 1
            hit.update(overlaps)
        else:
            fp += 1
    fn = len(gt_intervals) - len(hit)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, f1, tp, fp, fn


def safe_vus_pr(labels, scores, window_size):
    labels = np.asarray(labels).astype(bool)
    if get_vus_metrics is None or labels.sum() == 0:
        return np.nan
    try:
        out = get_vus_metrics(
            np.asarray(scores, dtype=float),
            labels.astype(int),
            metric="vus",
            version="opt",
            slidingWindow=int(window_size),
            thre=250,
        )
        return float(out.get("VUS_PR", np.nan))
    except Exception as exc:
        print(f"  VUS-PR failed: {exc}", flush=True)
        return np.nan


def aggregate_windows_to_timeline(window_values, starts, total_length, window_size):
    sums = np.zeros(total_length, dtype=np.float64)
    counts = np.zeros(total_length, dtype=np.float64)
    for value, start in zip(window_values, starts):
        sums[start:start + window_size] += value
        counts[start:start + window_size] += 1
    out = np.full(total_length, np.nan)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def aggregate_window_mask_fraction(window_mask, starts, total_length, window_size):
    hits = np.zeros(total_length, dtype=np.float64)
    counts = np.zeros(total_length, dtype=np.float64)
    for is_hit, start in zip(window_mask, starts):
        hits[start:start + window_size] += float(is_hit)
        counts[start:start + window_size] += 1
    out = np.full(total_length, np.nan)
    valid = counts > 0
    out[valid] = hits[valid] / counts[valid]
    return out


def discover_ucr_datasets(data_dir=DATA_DIR):
    return sorted(path.stem for path in Path(data_dir).glob("*.txt"))


def train_fm(
    train_seqs,
    amp_weights,
    raw_weights,
    residual_weights,
    raw_tail_mask,
    residual_tail_mask,
    kind,
    device,
    seed,
    iterations=30000,
    warmup_fraction=0.0,
    log_every=5000,
):
    dataset = TensorDataset(
        torch.tensor(train_seqs, dtype=torch.float, device=device),
        torch.tensor(amp_weights, dtype=torch.float, device=device),
        torch.tensor(raw_weights, dtype=torch.float, device=device),
        torch.tensor(residual_weights, dtype=torch.float, device=device),
        torch.tensor(raw_tail_mask, dtype=torch.bool, device=device),
        torch.tensor(residual_tail_mask, dtype=torch.bool, device=device),
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=128, sampler=RandomSampler(dataset, generator=gen))
    model = TemporalFM(train_seqs.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4000, gamma=0.9)
    matcher = TargetConditionalFlowMatcher(sigma=0.1)
    iterator = itertools.cycle(loader)
    warmup_steps = max(1, int(iterations * warmup_fraction))
    for step in range(1, iterations + 1):
        x1, amp_w, raw_w, residual_w, raw_tail_flag, residual_tail_flag = next(iterator)
        x0 = torch.randn_like(x1)
        t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
        vt = model(t, xt)
        per = ((vt - ut) ** 2).mean(dim=1)
        if kind == "balanced_raw_tail":
            tail_flag = raw_tail_flag
            body_flag = ~tail_flag
            if tail_flag.any() and body_flag.any():
                balanced_w = torch.empty_like(per)
                balanced_w[body_flag] = 0.5 * len(per) / body_flag.sum()
                balanced_w[tail_flag] = 0.5 * len(per) / tail_flag.sum()
            else:
                balanced_w = torch.ones_like(per)
            eta = min(1.0, step / warmup_steps)
            w = 1.0 + eta * (balanced_w - 1.0)
        elif kind == "balanced_residual_tail":
            tail_flag = residual_tail_flag
            body_flag = ~tail_flag
            if tail_flag.any() and body_flag.any():
                balanced_w = torch.empty_like(per)
                balanced_w[body_flag] = 0.5 * len(per) / body_flag.sum()
                balanced_w[tail_flag] = 0.5 * len(per) / tail_flag.sum()
            else:
                balanced_w = torch.ones_like(per)
            eta = min(1.0, step / warmup_steps)
            w = 1.0 + eta * (balanced_w - 1.0)
        elif kind == "amplitude":
            w = amp_w
        elif kind.startswith("raw_radius"):
            eta = min(1.0, step / warmup_steps)
            w = 1.0 + eta * (raw_w - 1.0)
        elif kind.startswith("residual_tail"):
            eta = min(1.0, step / warmup_steps)
            w = 1.0 + eta * (residual_w - 1.0)
        else:
            w = torch.ones_like(amp_w)
        loss = (per * w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        tail_flag = residual_tail_flag if kind.startswith(("residual_tail", "balanced_residual")) else raw_tail_flag
        if step == 1 or step % log_every == 0 or step == iterations:
            print(
                f"  FM {kind} {step:05d}/{iterations} loss={float(loss.detach().cpu()):.4f} "
                f"body={float(per[~tail_flag].mean().detach().cpu()) if (~tail_flag).any() else np.nan:.4f} "
                f"tail={float(per[tail_flag].mean().detach().cpu()) if tail_flag.any() else np.nan:.4f}",
                flush=True,
            )
    return model


@torch.no_grad()
def score_fm(model, seqs, device, eval_batch=10000, ode_steps=4, ode_method="rk4"):
    loader = DataLoader(TensorDataset(torch.tensor(seqs, dtype=torch.float, device=device)), batch_size=eval_batch)
    backward_times = torch.linspace(1, 0, ode_steps, device=device)
    forward_times = torch.linspace(0, 1, ode_steps, device=device)
    recs, nlls, latent_radius2 = [], [], []
    model.eval()
    for (x,) in loader:
        z = torchdiffeq.odeint(lambda t, state: model(t, state), x, backward_times, method=ode_method)[-1]
        xhat = torchdiffeq.odeint(lambda t, state: model(t, state), z, forward_times, method=ode_method)[-1]
        recs.append(((xhat - x) ** 2).mean(dim=1).cpu())
        latent_radius2.append(torch.sum(z ** 2, dim=1).cpu())
        nlls.append((0.5 * torch.sum(z ** 2, dim=1) + (seqs.shape[1] / 2) * np.log(2 * np.pi)).cpu())
    return torch.cat(recs).numpy(), torch.cat(nlls).numpy(), torch.cat(latent_radius2).numpy()


def run_one(
    file_name,
    seed,
    fm_iterations,
    raw_boosts=(),
    residual_boosts=(4.0,),
    warmup_fraction=0.0,
    include_baselines=True,
    include_amplitude=True,
    include_balanced_raw=False,
    include_balanced_residual=False,
    artifact_dir=None,
    calibration_size=4096,
    vae_latent_dim=24,
    vae_iterations=30000,
    log_every=5000,
    use_clipping=True,
    clip_value=10.0,
    eval_batch=10000,
    ode_steps=4,
    ode_method="rk4",
    data_dir=DATA_DIR,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    train_seqs, test_seqs, test_labels, test_starts, timestamp_labels, train_split, anomaly_start, anomaly_end, total_length = load_dataset(
        file_name,
        use_clipping=use_clipping,
        clip_value=clip_value,
        data_dir=data_dir,
    )
    vae = train_vae(train_seqs, device, seed, latent_dim=vae_latent_dim, iterations=vae_iterations, log_every=log_every)
    train_r2 = encode_arrays(vae, train_seqs, device)
    test_r2 = encode_arrays(vae, test_seqs, device)
    train_std = train_seqs.std(axis=1)
    test_std = test_seqs.std(axis=1)
    coef, *_ = np.linalg.lstsq(np.c_[np.ones_like(train_std), train_std], train_r2, rcond=None)
    train_resid = train_r2 - (np.c_[np.ones_like(train_std), train_std] @ coef)
    test_resid = test_r2 - (np.c_[np.ones_like(test_std), test_std] @ coef)
    common_thr, tail_thr = np.quantile(train_resid, [0.50, 0.90])
    raw_tail_thr = np.quantile(train_r2, 0.90)
    normal = ~test_labels
    common_mask = normal & (test_resid <= common_thr)
    tail_mask = normal & (test_resid >= tail_thr)
    amp_w = bounded_weights(train_std)
    calib_rng = np.random.default_rng(seed)
    calib_idx = calib_rng.choice(len(train_seqs), size=min(calibration_size, len(train_seqs)), replace=False)
    rows = []
    kinds = ["uniform"] if include_baselines else []
    if include_baselines and include_amplitude:
        kinds.append("amplitude")
    kinds.extend([f"raw_radius_b{boost:g}_w{warmup_fraction:g}" for boost in raw_boosts])
    kinds.extend([f"residual_tail_b{boost:g}_w{warmup_fraction:g}" for boost in residual_boosts])
    if include_balanced_raw:
        kinds.append("balanced_raw_tail")
    if include_balanced_residual:
        kinds.append("balanced_residual_tail")
    raw_weights_by_kind = {
        f"raw_radius_b{boost:g}_w{warmup_fraction:g}": bounded_weights(train_r2, boost=boost)
        for boost in raw_boosts
    }
    residual_weights_by_kind = {
        f"residual_tail_b{boost:g}_w{warmup_fraction:g}": bounded_weights(train_resid, boost=boost)
        for boost in residual_boosts
    }
    for kind in kinds:
        set_seed(seed)
        raw_w = raw_weights_by_kind.get(kind, bounded_weights(train_r2))
        residual_w = residual_weights_by_kind.get(kind, bounded_weights(train_resid))
        model = train_fm(
            train_seqs,
            amp_w,
            raw_w,
            residual_w,
            train_r2 >= raw_tail_thr,
            train_resid >= tail_thr,
            kind,
            device,
            seed,
            iterations=fm_iterations,
            warmup_fraction=warmup_fraction if kind.startswith(("raw_radius", "residual_tail", "balanced_")) else 0.0,
            log_every=log_every,
        )
        calib_rec, calib_nll, calib_latent_radius2 = score_fm(
            model,
            train_seqs[calib_idx],
            device,
            eval_batch=eval_batch,
            ode_steps=ode_steps,
            ode_method=ode_method,
        )
        rec, nll, latent_radius2 = score_fm(
            model,
            test_seqs,
            device,
            eval_batch=eval_batch,
            ode_steps=ode_steps,
            ode_method=ode_method,
        )
        window_metrics = detection_metrics(test_labels, nll)
        timeline_nll = aggregate_windows_to_timeline(nll, test_starts, total_length, train_seqs.shape[1])
        valid_timeline = ~np.isnan(timeline_nll)
        timestamp_tail_resid = aggregate_windows_to_timeline(test_resid, test_starts, total_length, train_seqs.shape[1])
        timestamp_tail_mask = valid_timeline & ~timestamp_labels & (timestamp_tail_resid >= tail_thr)
        timeline_metrics = detection_metrics(timestamp_labels[valid_timeline], timeline_nll[valid_timeline])
        window_fp90 = fp_at_recall(test_labels, nll, 0.90)
        window_fp95 = fp_at_recall(test_labels, nll, 0.95)
        tail_fp90 = fp_at_recall(test_labels, nll, 0.90, subset_mask=tail_mask)
        tail_fp95 = fp_at_recall(test_labels, nll, 0.95, subset_mask=tail_mask)
        timestamp_fp90 = fp_at_recall(timestamp_labels[valid_timeline], timeline_nll[valid_timeline], 0.90)
        timestamp_fp95 = fp_at_recall(timestamp_labels[valid_timeline], timeline_nll[valid_timeline], 0.95)
        timestamp_tail_fp90 = fp_at_recall(
            timestamp_labels[valid_timeline],
            timeline_nll[valid_timeline],
            0.90,
            subset_mask=timestamp_tail_mask[valid_timeline],
        )
        timestamp_tail_fp95 = fp_at_recall(
            timestamp_labels[valid_timeline],
            timeline_nll[valid_timeline],
            0.95,
            subset_mask=timestamp_tail_mask[valid_timeline],
        )
        window_pred95 = nll >= window_fp95["threshold"] if np.isfinite(window_fp95["threshold"]) else np.zeros_like(test_labels, dtype=bool)
        gt_window_intervals = merge_intervals(
            [(int(start), int(start + train_seqs.shape[1] - 1)) for start, flag in zip(test_starts, test_labels) if flag]
        )
        pred_window_intervals = merge_intervals(
            [(int(start), int(start + train_seqs.shape[1] - 1)) for start, flag in zip(test_starts, window_pred95) if flag]
        )
        hundman_precision, hundman_f1, hundman_tp, hundman_fp, hundman_fn = interval_overlap_metrics(
            gt_window_intervals,
            pred_window_intervals,
        )
        timestamp_pred95 = (
            timeline_nll[valid_timeline] >= timestamp_fp95["threshold"]
            if np.isfinite(timestamp_fp95["threshold"])
            else np.zeros_like(timestamp_labels[valid_timeline], dtype=bool)
        )
        event_precision, event_f1, event_tp, event_fp, event_fn = interval_overlap_metrics(
            ranges(timestamp_labels[valid_timeline]),
            ranges(timestamp_pred95),
        )
        rows.append({
            "dataset": file_name,
            "seed": seed,
            "model": kind,
            "n_common": int(common_mask.sum()),
            "n_tail": int(tail_mask.sum()),
            "n_anomaly": int(test_labels.sum()),
            "corr_std_raw_r2": float(np.corrcoef(test_std[normal], test_r2[normal])[0, 1]),
            "corr_std_resid": float(np.corrcoef(test_std[normal], test_resid[normal])[0, 1]),
            "common_rec": float(rec[common_mask].mean()),
            "tail_rec": float(rec[tail_mask].mean()),
            "anomaly_rec": float(rec[test_labels].mean()),
            "common_nll": float(nll[common_mask].mean()),
            "tail_nll": float(nll[tail_mask].mean()),
            "anomaly_nll": float(nll[test_labels].mean()),
            "anomaly_minus_tail_nll": float(nll[test_labels].mean() - nll[tail_mask].mean()),
            "window_auroc": window_metrics["auroc"],
            "window_auprc": window_metrics["auprc"],
            "window_best_f1": window_metrics["best_f1"],
            "timestamp_auroc": timeline_metrics["auroc"],
            "timestamp_auprc": timeline_metrics["auprc"],
            "timestamp_vus_pr": safe_vus_pr(timestamp_labels[valid_timeline], timeline_nll[valid_timeline], train_seqs.shape[1]),
            "timestamp_best_f1": timeline_metrics["best_f1"],
            "window_threshold_at_90_recall": window_fp90["threshold"],
            "window_achieved_recall_at_90": window_fp90["achieved_recall"],
            "window_fp_rate_at_90_recall": window_fp90["fp_rate"],
            "window_threshold_at_95_recall": window_fp95["threshold"],
            "window_achieved_recall_at_95": window_fp95["achieved_recall"],
            "window_fp_rate_at_95_recall": window_fp95["fp_rate"],
            "timestamp_threshold_at_90_recall": timestamp_fp90["threshold"],
            "timestamp_achieved_recall_at_90": timestamp_fp90["achieved_recall"],
            "timestamp_fp_rate_at_90_recall": timestamp_fp90["fp_rate"],
            "timestamp_threshold_at_95_recall": timestamp_fp95["threshold"],
            "timestamp_achieved_recall_at_95": timestamp_fp95["achieved_recall"],
            "timestamp_fp_rate_at_95_recall": timestamp_fp95["fp_rate"],
            "tail_window_fp_rate_at_90_recall": tail_fp90["fp_rate"],
            "tail_window_fp_rate_at_95_recall": tail_fp95["fp_rate"],
            "tail_timestamp_fp_rate_at_90_recall": timestamp_tail_fp90["fp_rate"],
            "tail_timestamp_fp_rate_at_95_recall": timestamp_tail_fp95["fp_rate"],
            "hundman_window_precision_at_95_recall": hundman_precision,
            "hundman_window_f1_at_95_recall": hundman_f1,
            "hundman_window_tp": hundman_tp,
            "hundman_window_fp": hundman_fp,
            "hundman_window_fn": hundman_fn,
            "event_precision_at_95_recall": event_precision,
            "event_f1_at_95_recall": event_f1,
            "event_tp": event_tp,
            "event_fp": event_fp,
            "event_fn": event_fn,
            "ode_method": ode_method,
            "ode_steps": ode_steps,
        })
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                artifact_dir / f"{file_name}__seed{seed}__{kind}.npz",
                nll=nll,
                rec=rec,
                latent_radius2=latent_radius2,
                calib_rec=calib_rec,
                calib_nll=calib_nll,
                calib_latent_radius2=calib_latent_radius2,
                calib_indices=calib_idx,
                test_labels=test_labels,
                tail_mask=tail_mask,
                common_mask=common_mask,
                test_starts=test_starts,
                timestamp_nll=timeline_nll,
                timestamp_labels=timestamp_labels,
                timestamp_tail_resid=timestamp_tail_resid,
                timestamp_tail_mask=timestamp_tail_mask,
                train_split=np.array(train_split),
                anomaly_start=np.array(anomaly_start),
                anomaly_end=np.array(anomaly_end),
                anomaly_length=np.array(anomaly_end - anomaly_start + 1),
                window_size=np.array(train_seqs.shape[1]),
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="residual_tail_fm_batch_results.csv")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing UCR .txt files.")
    parser.add_argument("--all-ucr", action="store_true")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--fm-iterations", type=int, default=30000)
    parser.add_argument("--raw-boosts", nargs="+", type=float, default=[])
    parser.add_argument("--residual-boosts", nargs="+", type=float, default=[4.0])
    parser.add_argument("--warmup-fraction", type=float, default=0.0)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--exclude-amplitude", action="store_true")
    parser.add_argument("--include-balanced-raw", action="store_true")
    parser.add_argument("--include-balanced-residual", action="store_true")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--calibration-size", type=int, default=20000)
    parser.add_argument("--vae-latent-dim", type=int, default=24)
    parser.add_argument("--vae-iterations", type=int, default=30000)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--use-clipping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-value", type=float, default=10.0)
    parser.add_argument("--eval-batch", type=int, default=10000)
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--ode-method", default="rk4")
    args = parser.parse_args()
    if args.dataset_file is not None:
        datasets = [
            line.strip()
            for line in Path(args.dataset_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif args.datasets is not None:
        datasets = args.datasets
    elif args.all_ucr:
        datasets = discover_ucr_datasets(args.data_dir)
    else:
        datasets = DATASETS
    if args.max_datasets is not None:
        datasets = datasets[:args.max_datasets]
    print(f"selected datasets: {len(datasets)}", flush=True)
    print(f"selected seeds: {args.seeds}", flush=True)
    out = Path(args.out)
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    completed = set(zip(existing.get("dataset", []), existing.get("seed", []))) if len(existing) else set()
    all_rows = existing.to_dict("records") if len(existing) else []
    for dataset in datasets:
        for seed in args.seeds:
            if (dataset, seed) in completed:
                print("skip", dataset, seed)
                continue
            print("run", dataset, seed, flush=True)
            rows = run_one(
                dataset,
                seed,
                args.fm_iterations,
                raw_boosts=tuple(args.raw_boosts),
                residual_boosts=tuple(args.residual_boosts),
                warmup_fraction=args.warmup_fraction,
                include_baselines=not args.skip_baselines,
                include_amplitude=not args.exclude_amplitude,
                include_balanced_raw=args.include_balanced_raw,
                include_balanced_residual=args.include_balanced_residual,
                artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
                calibration_size=args.calibration_size,
                vae_latent_dim=args.vae_latent_dim,
                vae_iterations=args.vae_iterations,
                log_every=args.log_every,
                use_clipping=args.use_clipping,
                clip_value=args.clip_value,
                eval_batch=args.eval_batch,
                ode_steps=args.ode_steps,
                ode_method=args.ode_method,
                data_dir=args.data_dir,
            )
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(out, index=False)
            print("saved", out, flush=True)


if __name__ == "__main__":
    main()
