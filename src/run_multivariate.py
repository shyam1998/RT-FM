"""Multivariate TSAD benchmark runner for RT-FM.

The runner preserves the channel-by-time window structure, trains one VAE per
dataset/seed cache entry, and then evaluates multiple Flow Matching weighting
variants. It supports the multivariate benchmark formats used in the paper:
GECCO/SWAN from DCdetector-style files, Anomaly Transformer-style MSL/SMAP/SMD,
and PSM CSV files.
"""

import argparse
import ast
import itertools
import json
import math
import random
import time
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


ROOT = Path("data/multivariate/anomaly_transformer_benchmarks")
DC_ROOT = Path("data/DCdetector_dataset")
NASA_ROOT = Path("data/multivariate/msl")
if not NASA_ROOT.exists():
    NASA_ROOT = Path("data/multivariate/MSL")


DATASETS = {
    "MSL": {"kind": "npy", "train": ROOT / "MSL/MSL/MSL_train.npy", "test": ROOT / "MSL/MSL/MSL_test.npy", "label": ROOT / "MSL/MSL/MSL_test_label.npy"},
    "SMAP": {"kind": "npy", "train": ROOT / "SMAP/SMAP/SMAP_train.npy", "test": ROOT / "SMAP/SMAP/SMAP_test.npy", "label": ROOT / "SMAP/SMAP/SMAP_test_label.npy"},
    "SMD": {"kind": "npy", "train": ROOT / "SMD/SMD/SMD_train.npy", "test": ROOT / "SMD/SMD/SMD_test.npy", "label": ROOT / "SMD/SMD/SMD_test_label.npy"},
    "PSM": {"kind": "csv", "train": ROOT / "PSM/PSM/train.csv", "test": ROOT / "PSM/PSM/test.csv", "label": ROOT / "PSM/PSM/test_label.csv"},
    "DC_GECCO": {"kind": "npy", "train": DC_ROOT / "NIPS_TS_GECCO/NIPS_TS_Water_train.npy", "test": DC_ROOT / "NIPS_TS_GECCO/NIPS_TS_Water_test.npy", "label": DC_ROOT / "NIPS_TS_GECCO/NIPS_TS_Water_test_label.npy"},
    "DC_SWAN": {"kind": "npy", "train": DC_ROOT / "NIPS_TS_Swan/NIPS_TS_Swan_train.npy", "test": DC_ROOT / "NIPS_TS_Swan/NIPS_TS_Swan_test.npy", "label": DC_ROOT / "NIPS_TS_Swan/NIPS_TS_Swan_test_label.npy"},
    "DC_CREDITCARD": {"kind": "npy", "train": DC_ROOT / "NIPS_TS_Creditcard/NIPS_TS_creditcard_train.npy", "test": DC_ROOT / "NIPS_TS_Creditcard/NIPS_TS_creditcard_test.npy", "label": DC_ROOT / "NIPS_TS_Creditcard/NIPS_TS_creditcard_test_label.npy"},
    "DC_SWAT": {"kind": "dc_swat", "train": DC_ROOT / "SWAT/swat_train2.csv", "test": DC_ROOT / "SWAT/swat2.csv"},
    "MSL_CHANNEL": {"kind": "nasa_channel", "root": NASA_ROOT, "spacecraft": "MSL"},
    "SMAP_CHANNEL": {"kind": "nasa_channel", "root": NASA_ROOT, "spacecraft": "SMAP"},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualBlock1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


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
    def __init__(self, num_channels, seq_len, latent_dim=24, hidden_size=64, num_blocks=3):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, hidden_size, kernel_size=3, padding="same"),
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
            nn.Conv1d(hidden_size, num_channels, kernel_size=3, padding="same"),
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
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h).clamp(-10.0, 10.0)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        h = self.decoder_input(z).view(z.shape[0], -1, self.seq_len)
        return self.decoder_blocks(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def vae_terms(x, x_hat, mu, logvar):
    rec = ((x_hat - x) ** 2).mean(dim=(1, 2))
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
        group_count = valid_group_count(channels, num_groups)
        self.block = nn.Sequential(
            nn.GroupNorm(group_count, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
            nn.GroupNorm(group_count, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
        )

    def forward(self, x):
        return x + self.block(x)


class TemporalFM(nn.Module):
    def __init__(self, num_channels, seq_len, hidden_size=128, num_layers=5, time_embed_dim=None, num_groups=8):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.time_embed_dim = time_embed_dim or hidden_size
        self.raw_time_dim = min(64, self.time_embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.raw_time_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.input_conv = nn.Conv1d(num_channels, hidden_size, kernel_size=3, padding="same")
        self.cond_proj = nn.Linear(self.time_embed_dim + 2 * num_channels, hidden_size)
        self.res_blocks = nn.Sequential(*[FMResidualBlock(hidden_size, num_groups=num_groups) for _ in range(num_layers)])
        self.output_conv = nn.Conv1d(hidden_size, num_channels, kernel_size=3, padding="same")
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
        channel_features = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        cond = self.cond_proj(torch.cat([time_emb, channel_features], dim=1))[:, :, None]
        h = self.input_conv(x) + cond
        h = self.res_blocks(h)
        return self.output_conv(h)


def dataset_available(name, spec):
    if spec["kind"] in {"npy", "csv"}:
        return all(spec[k].exists() for k in ["train", "test", "label"])
    if spec["kind"] == "dc_swat":
        return spec["train"].exists() and spec["test"].exists()
    if spec["kind"] == "nasa_channel":
        return (spec["root"] / "labeled_anomalies.csv").exists() and (spec["root"] / "data/data/train").exists()
    return False


def standard_scale_train_test(train, test, kind=None, use_clipping=True, clip_value=10.0):
    train_nan_frac = float(np.isnan(train).mean())
    test_nan_frac = float(np.isnan(test).mean())
    scale_floor = 1e-3
    robust_scaled = kind == "nasa_channel"
    if robust_scaled:
        center = np.nanmedian(train, axis=0, keepdims=True)
        q25 = np.nanquantile(train, 0.25, axis=0, keepdims=True)
        q75 = np.nanquantile(train, 0.75, axis=0, keepdims=True)
        scale = q75 - q25
    else:
        center = np.nanmean(train, axis=0, keepdims=True)
        scale = np.nanstd(train, axis=0, keepdims=True)
    scale = np.where(scale > scale_floor, scale, scale_floor)
    train_z = np.nan_to_num((train - center) / scale).astype(np.float32)
    test_z = np.nan_to_num((test - center) / scale).astype(np.float32)
    if use_clipping or robust_scaled:
        train_z = np.clip(train_z, -clip_value, clip_value).astype(np.float32)
        test_z = np.clip(test_z, -clip_value, clip_value).astype(np.float32)
    diagnostics = {
        "train_nan_frac": train_nan_frac,
        "test_nan_frac": test_nan_frac,
        "scale_min": float(scale.min()),
        "scale_median": float(np.median(scale)),
        "scale_max": float(scale.max()),
        "train_abs_max": float(np.abs(train_z).max()),
        "train_abs_q999": float(np.quantile(np.abs(train_z), 0.999)),
        "test_abs_max": float(np.abs(test_z).max()),
        "test_abs_q999": float(np.quantile(np.abs(test_z), 0.999)),
        "scale_floor": scale_floor,
        "robust_scaled": robust_scaled,
        "use_clipping": bool(use_clipping or robust_scaled),
        "clip_value": float(clip_value),
    }
    return train_z, test_z, diagnostics


def load_nasa_channel(spec, channel_id):
    root = spec["root"]
    train = np.load(root / "data/data/train" / f"{channel_id}.npy", allow_pickle=True).astype(np.float32)
    test = np.load(root / "data/data/test" / f"{channel_id}.npy", allow_pickle=True).astype(np.float32)
    meta = pd.read_csv(root / "labeled_anomalies.csv")
    row = meta[(meta["chan_id"] == channel_id) & (meta["spacecraft"] == spec["spacecraft"])]
    labels = np.zeros(len(test), dtype=bool)
    if not row.empty:
        for a, b in ast.literal_eval(row.iloc[0]["anomaly_sequences"]):
            labels[max(0, int(a)) : min(len(test), int(b) + 1)] = True
    return train, test, labels, None


def load_dataset(name, channel_id, use_clipping=False, clip_value=10.0, drop_train_anomalies=True):
    spec = DATASETS[name]
    kind = spec["kind"]
    train_labels = None
    if kind == "npy":
        train = np.load(spec["train"], allow_pickle=True).astype(np.float32)
        test = np.load(spec["test"], allow_pickle=True).astype(np.float32)
        labels = np.load(spec["label"], allow_pickle=True).astype(bool).reshape(-1)
    elif kind == "csv":
        train_df = pd.read_csv(spec["train"])
        test_df = pd.read_csv(spec["test"])
        label_df = pd.read_csv(spec["label"])
        train = train_df.drop(columns=[c for c in train_df.columns if "timestamp" in c.lower()], errors="ignore").to_numpy(np.float32)
        test = test_df.drop(columns=[c for c in test_df.columns if "timestamp" in c.lower()], errors="ignore").to_numpy(np.float32)
        label_col = [c for c in label_df.columns if "label" in c.lower()]
        labels = (label_df[label_col[0]] if label_col else label_df.iloc[:, -1]).to_numpy().astype(bool)
    elif kind == "dc_swat":
        train_df = pd.read_csv(spec["train"])
        test_df = pd.read_csv(spec["test"])
        label_col = "Normal/Attack"
        train_labels = train_df[label_col].to_numpy().astype(bool) if label_col in train_df.columns else np.zeros(len(train_df), dtype=bool)
        labels = test_df[label_col].to_numpy().astype(bool)
        drop_cols = [c for c in train_df.columns if c.strip().lower() in ["normal/attack", "timestamp", "time"]]
        train = train_df.drop(columns=drop_cols, errors="ignore").to_numpy(np.float32)
        test = test_df.drop(columns=drop_cols, errors="ignore").to_numpy(np.float32)
    elif kind == "nasa_channel":
        train, test, labels, train_labels = load_nasa_channel(spec, channel_id)
    else:
        raise ValueError(kind)
    if train.ndim == 1:
        train = train[:, None]
    if test.ndim == 1:
        test = test[:, None]
    if train_labels is not None and drop_train_anomalies:
        train_labels = np.asarray(train_labels).astype(bool)[: len(train)]
        train = train[~train_labels]
    n = min(len(test), len(labels))
    test, labels = test[:n], labels[:n]
    train, test, diag = standard_scale_train_test(train, test, kind=kind, use_clipping=use_clipping, clip_value=clip_value)
    return train, test, labels, diag


def make_windows(x, labels=None, window_size=64, stride=1):
    starts = np.arange(0, len(x) - window_size + 1, stride, dtype=np.int64)
    windows = np.stack([x[s : s + window_size].T for s in starts]).astype(np.float32)
    win_labels = np.zeros(len(starts), dtype=bool) if labels is None else np.array([labels[s : s + window_size].any() for s in starts], bool)
    return windows, starts, win_labels


def residualize_against_channel_amplitude(radius2, amp_features, ridge=1e-3):
    x = np.c_[np.ones(len(amp_features)), amp_features]
    reg = ridge * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + reg, x.T @ radius2)
    return radius2 - x @ coef, coef


def apply_channel_amplitude_residual(radius2, amp_features, coef):
    return radius2 - np.c_[np.ones(len(amp_features)), amp_features] @ coef


def bounded_weights(scores, boost=4.0):
    lo, hi = np.quantile(scores, [0.90, 0.99])
    strength = np.clip((scores - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    w = 1.0 + boost * strength
    return (w / w.mean()).astype(np.float32)


def parse_boost(kind, default=1.0):
    if "_b" not in kind:
        return default
    try:
        return float(kind.split("_b", 1)[1].split("_", 1)[0])
    except Exception:
        return default


def make_weights(kind, train_amp, train_radius2, train_resid, q_low=0.90):
    n = len(train_resid)
    train_tail_mask = train_resid >= np.quantile(train_resid, q_low)
    if kind == "uniform":
        return np.ones(n, dtype=np.float32), np.zeros(n, dtype=bool)
    if kind == "amplitude":
        return bounded_weights(train_amp, boost=4.0), train_amp >= np.quantile(train_amp, q_low)
    if kind.startswith("raw_radius"):
        return bounded_weights(train_radius2, boost=parse_boost(kind)), train_radius2 >= np.quantile(train_radius2, q_low)
    if kind.startswith("residual_tail"):
        return bounded_weights(train_resid, boost=parse_boost(kind)), train_tail_mask
    if kind == "balanced_raw_tail":
        return np.ones(n, dtype=np.float32), train_radius2 >= np.quantile(train_radius2, q_low)
    if kind == "balanced_residual_tail":
        return np.ones(n, dtype=np.float32), train_tail_mask
    raise ValueError(kind)


def train_vae(train_seqs, cfg, device):
    set_seed(cfg.seed)
    dataset = TensorDataset(torch.tensor(train_seqs, dtype=torch.float, device=device))
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.vae_batch, sampler=RandomSampler(dataset, generator=gen))
    iterator = itertools.cycle(loader)
    model = TemporalVAE(train_seqs.shape[1], cfg.window_size, latent_dim=cfg.vae_latent_dim, hidden_size=cfg.vae_hidden, num_blocks=3).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.vae_lr, weight_decay=1e-4)
    model.train()
    t0 = time.time()
    last = {}
    for step in range(1, cfg.vae_iterations + 1):
        (x,) = next(iterator)
        opt.zero_grad(set_to_none=True)
        x_hat, mu, logvar = model(x)
        rec, kl = vae_terms(x, x_hat, mu, logvar)
        loss = rec.mean() + cfg.vae_beta * kl.mean()
        loss.backward()
        opt.step()
        if step == 1 or step % cfg.log_every == 0 or step == cfg.vae_iterations:
            last = {"vae_loss": float(loss.detach().cpu()), "vae_rec": float(rec.mean().detach().cpu()), "vae_kl": float(kl.mean().detach().cpu()), "vae_elapsed_s": time.time() - t0}
            print(f"  VAE {step:05d}/{cfg.vae_iterations} loss={last['vae_loss']:.4f} rec={last['vae_rec']:.4f} kl={last['vae_kl']:.4f}", flush=True)
    return model, last


@torch.no_grad()
def encode_all(model, windows, device, batch=4096):
    model.eval()
    mus, logvars, recs = [], [], []
    loader = DataLoader(TensorDataset(torch.tensor(windows, dtype=torch.float, device=device)), batch_size=batch)
    for (x,) in loader:
        x_hat, mu, logvar = model(x)
        rec, _ = vae_terms(x, x_hat, mu, logvar)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
        recs.append(rec.cpu())
    return torch.cat(mus).numpy(), torch.cat(logvars).numpy(), torch.cat(recs).numpy()


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def vae_cache_id(dataset_name, seed, cfg):
    channel = cfg.channel_id if dataset_name.endswith("_CHANNEL") else "all"
    parts = [
        dataset_name,
        channel,
        f"seed{seed}",
        f"w{cfg.window_size}",
        f"tr{cfg.train_stride}",
        f"te{cfg.test_stride}",
        f"ld{cfg.vae_latent_dim}",
        f"vh{cfg.vae_hidden}",
        f"vi{cfg.vae_iterations}",
        f"beta{cfg.vae_beta}",
        f"clip{int(cfg.use_clipping)}_{cfg.clip_value}",
        f"drop{int(cfg.drop_train_anomalies)}",
    ]
    return safe_name("_".join(map(str, parts)))


def load_or_build_vae_cache(dataset_name, seed, train_seqs, test_seqs, cfg, device):
    cache_dir = cfg.output_dir / "_vae_cache" / vae_cache_id(dataset_name, seed, cfg)
    cache_file = cache_dir / "vae_latents.npz"
    diag_file = cache_dir / "vae_diag.json"
    if cache_file.exists() and diag_file.exists() and not cfg.overwrite_vae_cache:
        print(f"  VAE cache hit: {cache_dir}", flush=True)
        data = np.load(cache_file)
        with open(diag_file) as f:
            vae_diag = json.load(f)
        return (
            data["train_mu"],
            data["train_logvar"],
            data["train_vae_rec"],
            data["test_mu"],
            data["test_logvar"],
            data["test_vae_rec"],
            vae_diag,
        )

    print(f"  VAE cache miss: training once for dataset/seed", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vae, vae_diag = train_vae(train_seqs, cfg, device)
    train_mu, train_logvar, train_vae_rec = encode_all(vae, train_seqs, device, cfg.eval_batch)
    test_mu, test_logvar, test_vae_rec = encode_all(vae, test_seqs, device, cfg.eval_batch)
    np.savez_compressed(
        cache_file,
        train_mu=train_mu,
        train_logvar=train_logvar,
        train_vae_rec=train_vae_rec,
        test_mu=test_mu,
        test_logvar=test_logvar,
        test_vae_rec=test_vae_rec,
    )
    with open(diag_file, "w") as f:
        json.dump(vae_diag, f, indent=2)
    return train_mu, train_logvar, train_vae_rec, test_mu, test_logvar, test_vae_rec, vae_diag


def train_fm(train_seqs, weights, group, model_kind, cfg, device):
    set_seed(cfg.seed)
    dataset = TensorDataset(
        torch.tensor(train_seqs, dtype=torch.float, device=device),
        torch.tensor(weights, dtype=torch.float, device=device),
        torch.tensor(group, dtype=torch.bool, device=device),
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.fm_batch, sampler=RandomSampler(dataset, generator=gen))
    iterator = itertools.cycle(loader)
    model = TemporalFM(train_seqs.shape[1], cfg.window_size, hidden_size=cfg.fm_hidden, num_layers=5).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.fm_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4000, gamma=0.9)
    matcher = TargetConditionalFlowMatcher(sigma=cfg.fm_sigma)
    warmup_steps = max(1, int(cfg.fm_iterations * cfg.warmup_fraction))
    t0 = time.time()
    last = {}
    for step in range(1, cfg.fm_iterations + 1):
        x1, base_w, tail_flag = next(iterator)
        x0 = torch.randn_like(x1)
        t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
        vt = model(t, xt)
        per = ((vt - ut) ** 2).mean(dim=(1, 2))
        if model_kind in {"balanced_raw_tail", "balanced_residual_tail"}:
            body_flag = ~tail_flag
            eff_w = torch.ones_like(per)
            if tail_flag.any() and body_flag.any():
                eff_w[body_flag] = 0.5 * len(per) / body_flag.sum()
                eff_w[tail_flag] = 0.5 * len(per) / tail_flag.sum()
        else:
            eff_w = base_w
        eta = min(1.0, step / warmup_steps)
        w = 1.0 + eta * (eff_w - 1.0)
        loss = (per * w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step == 1 or step % cfg.log_every == 0 or step == cfg.fm_iterations:
            last = {
                "fm_loss": float(loss.detach().cpu()),
                "fm_body_loss": float(per[~tail_flag].mean().detach().cpu()) if (~tail_flag).any() else np.nan,
                "fm_tail_loss": float(per[tail_flag].mean().detach().cpu()) if tail_flag.any() else np.nan,
                "fm_elapsed_s": time.time() - t0,
            }
            print(f"  FM {step:05d}/{cfg.fm_iterations} loss={last['fm_loss']:.4f} body={last['fm_body_loss']:.4f} tail={last['fm_tail_loss']:.4f}", flush=True)
    return model, last


@torch.no_grad()
def reconstruct(model, windows, cfg, device):
    model.eval()
    rec, nll, r2, channel_rec = [], [], [], []
    backward_times = torch.linspace(1, 0, cfg.ode_steps, device=device)
    forward_times = torch.linspace(0, 1, cfg.ode_steps, device=device)
    state_dim = windows.shape[1] * windows.shape[2]
    const = 0.5 * state_dim * np.log(2 * np.pi)
    loader = DataLoader(TensorDataset(torch.tensor(windows, dtype=torch.float, device=device)), batch_size=cfg.eval_batch)
    for (x,) in loader:
        z = torchdiffeq.odeint(lambda t, state: model(t, state), x, backward_times, method=cfg.ode_method)[-1]
        xhat = torchdiffeq.odeint(lambda t, state: model(t, state), z, forward_times, method=cfg.ode_method)[-1]
        rr = torch.sum(z ** 2, dim=(1, 2))
        err = (xhat - x) ** 2
        r2.append(rr.cpu())
        nll.append((0.5 * rr + const).cpu())
        rec.append(err.mean(dim=(1, 2)).cpu())
        channel_rec.append(err.mean(dim=2).cpu())
    return {"nll": torch.cat(nll).numpy(), "rec": torch.cat(rec).numpy(), "radius2": torch.cat(r2).numpy(), "channel_rec": torch.cat(channel_rec).numpy()}


def robust(s, ref, floor=1e-3):
    q25, q75 = np.quantile(ref, [0.25, 0.75])
    return (s - np.median(ref)) / max(q75 - q25, floor)


def auroc(y, s):
    y = np.asarray(y).astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    return float(roc_auc_score(y.astype(int), np.asarray(s, dtype=float)))


def auprc(y, s):
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return np.nan
    return float(average_precision_score(y.astype(int), np.asarray(s, dtype=float)))


def best_f1(y, s):
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-np.asarray(s, dtype=float))
    yy = y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(~yy)
    fn = yy.sum() - tp
    pr = tp / np.maximum(tp + fp, 1)
    rc = tp / np.maximum(tp + fn, 1)
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return float(np.nanmax(f))


def threshold_recall(y, s, target=0.95):
    y = np.asarray(y, bool)
    s = np.asarray(s, dtype=float)
    if y.sum() == 0:
        return np.nan, np.nan, np.nan, np.zeros_like(y, dtype=bool)
    for th in np.unique(s)[::-1]:
        pred = s >= th
        rc = (pred & y).sum() / max(y.sum(), 1)
        if rc >= target:
            return float(th), float(rc), float((pred & ~y).sum() / max((~y).sum(), 1)), pred
    return np.nan, np.nan, np.nan, np.zeros_like(y, dtype=bool)


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


def aggregate(scores, test_starts, test_len, window_size):
    sums = np.zeros(test_len, dtype=float)
    counts = np.zeros(test_len, dtype=float)
    for st, sc in zip(test_starts, scores):
        sums[st : st + window_size] += sc
        counts[st : st + window_size] += 1
    out = np.full(test_len, np.nan, dtype=float)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out, valid


def aggregate_window_mask_to_timestamps(mask, test_starts, test_len, window_size):
    counts = np.zeros(test_len, dtype=float)
    for st, flag in zip(test_starts, mask):
        if flag:
            counts[st : st + window_size] += 1
    return counts > 0


def aggregate_residual_tail_to_timestamps(test_resid, test_starts, test_len, window_size, threshold):
    resid_ts, valid = aggregate(test_resid, test_starts, test_len, window_size)
    out = np.zeros(test_len, dtype=bool)
    out[valid] = resid_ts[valid] >= threshold
    return out


def safe_vus_pr(y, s, cfg):
    if get_vus_metrics is None or np.asarray(y).sum() == 0:
        return np.nan
    try:
        out = get_vus_metrics(np.asarray(s, dtype=float), np.asarray(y).astype(int), metric="vus", version=cfg.vus_version, slidingWindow=cfg.window_size, thre=cfg.vus_thresholds)
        return float(out.get("VUS_PR", np.nan))
    except Exception as exc:
        print(f"  VUS-PR failed: {exc}", flush=True)
        return np.nan


def score_metrics(scores, calib, test_labels, timestamp_labels, test_starts, common_mask, rare_mask, rare_mask_q95, test_resid, rare_threshold_q90, rare_threshold_q95, cfg):
    nll = robust(scores["nll"], calib["nll"])
    rec = robust(scores["rec"], calib["rec"])
    families = {"NLL": nll, "reconstruction": rec, f"fused_{cfg.fusion_alpha:.2f}": cfg.fusion_alpha * nll + (1 - cfg.fusion_alpha) * rec}
    rows = []
    ts_rows = []
    rare_ts = aggregate_residual_tail_to_timestamps(test_resid, test_starts, len(timestamp_labels), cfg.window_size, rare_threshold_q90)
    rare_ts95 = aggregate_residual_tail_to_timestamps(test_resid, test_starts, len(timestamp_labels), cfg.window_size, rare_threshold_q95)
    for name, s in families.items():
        th, rc, fp, pred = threshold_recall(test_labels, s, 0.95)
        gt_intervals = merge_intervals([(int(st), int(st + cfg.window_size - 1)) for st, flag in zip(test_starts, test_labels) if flag])
        pred_intervals = merge_intervals([(int(st), int(st + cfg.window_size - 1)) for st, flag in zip(test_starts, pred) if flag])
        hp, hf, htp, hfp, hfn = interval_overlap_metrics(gt_intervals, pred_intervals)
        rows.append({
            "score": name,
            "win_AUROC": auroc(test_labels, s),
            "win_AP_AUPRC": auprc(test_labels, s),
            "win_best_F1": best_f1(test_labels, s),
            "win_FP_rate@95R": fp,
            "win_tail_q90_FP_rate@95R": (pred & rare_mask).sum() / max(rare_mask.sum(), 1),
            "win_tail_q95_FP_rate@95R": (pred & rare_mask_q95).sum() / max(rare_mask_q95.sum(), 1),
            "common_mean": np.nanmean(s[common_mask]) if common_mask.any() else np.nan,
            "rare_q90_mean": np.nanmean(s[rare_mask]) if rare_mask.any() else np.nan,
            "rare_q95_mean": np.nanmean(s[rare_mask_q95]) if rare_mask_q95.any() else np.nan,
            "anom_mean": np.nanmean(s[test_labels]) if test_labels.any() else np.nan,
            "hundman_window_precision@95R": hp,
            "hundman_window_F1@95R": hf,
            "hundman_window_TP": htp,
            "hundman_window_FP": hfp,
            "hundman_window_FN": hfn,
        })
        ts, valid = aggregate(s, test_starts, len(timestamp_labels), cfg.window_size)
        y = timestamp_labels[valid]
        ss = ts[valid]
        _, _, ts_fp, ts_pred = threshold_recall(y, ss, 0.95)
        p, f, tp, efp, fn = interval_overlap_metrics(ranges(y), ranges(ts_pred))
        ts_rows.append({
            "score": name,
            "ts_AUROC": auroc(y, ss),
            "ts_AP_AUPRC": auprc(y, ss),
            "ts_VUS_PR": safe_vus_pr(y, ss, cfg),
            "ts_best_F1": best_f1(y, ss),
            "ts_FP_rate@95R": ts_fp,
            "ts_tail_q90_FP_rate@95R": (ts_pred & rare_ts[valid]).sum() / max(rare_ts[valid].sum(), 1),
            "ts_tail_q95_FP_rate@95R": (ts_pred & rare_ts95[valid]).sum() / max(rare_ts95[valid].sum(), 1),
            "event_precision@95R": p,
            "event_F1@95R": f,
            "event_TP": tp,
            "event_FP": efp,
            "event_FN": fn,
        })
    return pd.DataFrame(rows), pd.DataFrame(ts_rows), {"nll_robust": nll, "rec_robust": rec, "fused": families[f"fused_{cfg.fusion_alpha:.2f}"]}


def run_one(dataset_name, model_kind, seed, cfg):
    cfg.seed = seed
    channel_id = cfg.channel_id
    run_id = f"{dataset_name.lower()}_{channel_id}_" if dataset_name.endswith("_CHANNEL") else f"{dataset_name.lower()}_"
    run_id += f"{model_kind}_seed{seed}"
    out_dir = cfg.output_dir / run_id
    done_file = out_dir / "done.json"
    if done_file.exists() and not cfg.overwrite:
        print(f"SKIP {run_id}", flush=True)
        return pd.read_csv(out_dir / "summary.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    print(f"\nRUN {run_id} on {device}", flush=True)
    train_raw, test_raw, labels, prep_diag = load_dataset(dataset_name, channel_id, cfg.use_clipping, cfg.clip_value, cfg.drop_train_anomalies)
    train_seqs, _, _ = make_windows(train_raw, None, cfg.window_size, cfg.train_stride)
    test_seqs, test_starts, test_labels = make_windows(test_raw, labels, cfg.window_size, cfg.test_stride)
    print(f"  train {train_raw.shape} windows {train_seqs.shape}; test {test_raw.shape} windows {test_seqs.shape}; anomaly windows {test_labels.mean():.4f}", flush=True)
    train_mu, train_logvar, train_vae_rec, test_mu, test_logvar, test_vae_rec, vae_diag = load_or_build_vae_cache(
        dataset_name, seed, train_seqs, test_seqs, cfg, device
    )
    train_r2 = (train_mu ** 2 + np.exp(train_logvar)).sum(axis=1)
    test_r2 = (test_mu ** 2 + np.exp(test_logvar)).sum(axis=1)
    train_amp_ch = train_seqs.std(axis=2)
    test_amp_ch = test_seqs.std(axis=2)
    train_amp = np.linalg.norm(train_amp_ch, axis=1) / np.sqrt(train_seqs.shape[1])
    train_resid, coef = residualize_against_channel_amplitude(train_r2, train_amp_ch)
    test_resid = apply_channel_amplitude_residual(test_r2, test_amp_ch, coef)
    lo90 = np.quantile(train_resid, cfg.q_low)
    lo95 = np.quantile(train_resid, 0.95)
    common_mask = (~test_labels) & (test_resid < np.median(train_resid))
    rare_mask = (~test_labels) & (test_resid >= lo90)
    rare_mask_q95 = (~test_labels) & (test_resid >= lo95)
    weights, group = make_weights(model_kind, train_amp, train_r2, train_resid, cfg.q_low)
    fm, fm_diag = train_fm(train_seqs, weights, group, model_kind, cfg, device)
    rng = np.random.default_rng(seed)
    calib_idx = rng.choice(len(train_seqs), size=min(cfg.calib_size, len(train_seqs)), replace=False)
    calib = reconstruct(fm, train_seqs[calib_idx], cfg, device)
    scores = reconstruct(fm, test_seqs, cfg, device)
    win_df, ts_df, robust_scores = score_metrics(scores, calib, test_labels, labels, test_starts, common_mask, rare_mask, rare_mask_q95, test_resid, lo90, lo95, cfg)
    win_df.insert(0, "level", "window")
    ts_df.insert(0, "level", "timestamp")
    summary = pd.concat([win_df, ts_df], ignore_index=True)
    summary.insert(0, "dataset", dataset_name)
    summary.insert(1, "model", model_kind)
    summary.insert(2, "seed", seed)
    summary["num_channels"] = train_seqs.shape[1]
    summary["train_windows"] = len(train_seqs)
    summary["test_windows"] = len(test_seqs)
    summary["test_anomaly_rate"] = float(labels.mean())
    summary["test_window_anomaly_rate"] = float(test_labels.mean())
    summary["test_rare_q90_frac"] = float(rare_mask.sum() / max((~test_labels).sum(), 1))
    summary["test_rare_q95_frac"] = float(rare_mask_q95.sum() / max((~test_labels).sum(), 1))
    summary["raw_radius_amp_corr"] = float(np.corrcoef(train_r2, train_amp)[0, 1])
    summary["resid_amp_corr"] = float(np.corrcoef(train_resid, train_amp)[0, 1])
    summary["train_resid_q50"] = float(np.quantile(train_resid, 0.50))
    summary["train_resid_q90"] = float(np.quantile(train_resid, 0.90))
    summary["train_resid_q95"] = float(np.quantile(train_resid, 0.95))
    summary["train_resid_q99"] = float(np.quantile(train_resid, 0.99))
    normal_test_mask = ~test_labels
    if normal_test_mask.any():
        summary["test_normal_resid_q50"] = float(np.quantile(test_resid[normal_test_mask], 0.50))
        summary["test_normal_resid_q90"] = float(np.quantile(test_resid[normal_test_mask], 0.90))
        summary["test_normal_resid_q95"] = float(np.quantile(test_resid[normal_test_mask], 0.95))
        summary["test_normal_resid_q99"] = float(np.quantile(test_resid[normal_test_mask], 0.99))
    else:
        summary["test_normal_resid_q50"] = np.nan
        summary["test_normal_resid_q90"] = np.nan
        summary["test_normal_resid_q95"] = np.nan
        summary["test_normal_resid_q99"] = np.nan
    for k, v in {**prep_diag, **vae_diag, **fm_diag}.items():
        summary[k] = v
    summary.to_csv(out_dir / "summary.csv", index=False)
    win_df.to_csv(out_dir / "window_metrics.csv", index=False)
    ts_df.to_csv(out_dir / "timestamp_metrics.csv", index=False)
    np.savez_compressed(
        out_dir / "artifacts.npz",
        test_starts=test_starts,
        timestamp_labels=labels,
        test_window_labels=test_labels,
        common_mask=common_mask,
        rare_mask=rare_mask,
        rare_mask_q95=rare_mask_q95,
        nll=scores["nll"],
        rec=scores["rec"],
        nll_robust=robust_scores["nll_robust"],
        rec_robust=robust_scores["rec_robust"],
        fused=robust_scores["fused"],
        train_resid=train_resid,
        test_resid=test_resid,
        train_radius2=train_r2,
        test_radius2=test_r2,
        fm_channel_rec=scores["channel_rec"],
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(cfg) | {"dataset": dataset_name, "model": model_kind, "seed": seed}, f, indent=2, default=str)
    with open(done_file, "w") as f:
        json.dump({"done": True, "time": time.time()}, f)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["DC_GECCO", "DC_SWAN", "MSL", "SMAP", "PSM", "SMD"])
    parser.add_argument("--models", nargs="+", default=["uniform", "residual_tail_b1_w0.2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--channel-id", default="A-1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multivariate_batch"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--vae-iterations", type=int, default=30000)
    parser.add_argument("--fm-iterations", type=int, default=30000)
    parser.add_argument("--vae-latent-dim", type=int, default=24)
    parser.add_argument("--vae-hidden", type=int, default=64)
    parser.add_argument("--fm-hidden", type=int, default=128)
    parser.add_argument("--vae-batch", type=int, default=128)
    parser.add_argument("--fm-batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=10000)
    parser.add_argument("--calib-size", type=int, default=20000)
    parser.add_argument("--vae-beta", type=float, default=0.05)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--fm-lr", type=float, default=1e-4)
    parser.add_argument("--fm-sigma", type=float, default=0.1)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--q-low", type=float, default=0.90)
    parser.add_argument("--q-high", type=float, default=0.99)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--ode-method", default="rk4")
    parser.add_argument("--fusion-alpha", type=float, default=0.5)
    parser.add_argument("--vus-thresholds", type=int, default=50)
    parser.add_argument("--vus-version", default="opt")
    parser.add_argument("--use-clipping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-value", type=float, default=10.0)
    parser.add_argument("--drop-train-anomalies", action="store_true", default=True)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-vae-cache", action="store_true")
    return parser.parse_args()


def main():
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in cfg.datasets:
        if dataset not in DATASETS:
            print(f"UNKNOWN DATASET {dataset}; skipping", flush=True)
            continue
        if not dataset_available(dataset, DATASETS[dataset]):
            print(f"UNAVAILABLE {dataset}; skipping", flush=True)
            continue
        for seed in cfg.seeds:
            for model in cfg.models:
                try:
                    rows.append(run_one(dataset, model, seed, cfg))
                    pd.concat(rows, ignore_index=True).to_csv(cfg.output_dir / "multivariate_batch_results.csv", index=False)
                except Exception as exc:
                    err = {"dataset": dataset, "model": model, "seed": seed, "error": repr(exc)}
                    pd.DataFrame([err]).to_csv(cfg.output_dir / f"error_{dataset}_{model}_seed{seed}.csv", index=False)
                    print(f"ERROR {err}", flush=True)
    if rows:
        all_results = pd.concat(rows, ignore_index=True)
        all_results.to_csv(cfg.output_dir / "multivariate_batch_results.csv", index=False)
        metric_cols = [c for c in all_results.columns if c not in {"dataset", "model", "seed", "level", "score"} and pd.api.types.is_numeric_dtype(all_results[c])]
        all_results.groupby(["level", "score", "model"])[metric_cols].mean(numeric_only=True).reset_index().to_csv(cfg.output_dir / "multivariate_batch_summary.csv", index=False)
        print("saved", cfg.output_dir / "multivariate_batch_results.csv", flush=True)


if __name__ == "__main__":
    main()
