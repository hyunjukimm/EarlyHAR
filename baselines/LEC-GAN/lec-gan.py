# lec_gan_with_halting.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utils
# =========================
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mse(x_hat: torch.Tensor, x_true: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes MSE only on positions where mask == 0 (imputed positions)
    x_hat, x_true: (B, T, V)
    mask: (B, T, V) where 1 means observed, 0 means missing
    """
    miss = (1.0 - mask)
    num = (miss.sum() + eps)
    return (((x_hat - x_true) ** 2) * miss).sum() / num


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def macro_f1(pred: torch.Tensor, y: torch.Tensor, n_classes: int, eps: float = 1e-12) -> float:
    pred = pred.view(-1)
    y = y.view(-1)
    f1s = []
    for c in range(n_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        f1s.append(f1)
    return float(sum(f1s) / max(1, n_classes))


def harmonic_mean_f1_earliness(f1: float, el: float, eps: float = 1e-12) -> float:
    """
    Timeliness score = 1 - EL
    HM = 2 * F1 * (1-EL) / (F1 + (1-EL))
    """
    t = 1.0 - el
    return float((2.0 * f1 * t) / (f1 + t + eps))


# =========================
# Masking for early setting
# =========================
@dataclass
class EarlyMaskConfig:
    """
    For each sample, we simulate an "early prefix" of length tau (in steps),
    and hide the rest as missing.
    base_missing_mask is the original missingness (if you have it).
    """
    T: int
    # you can evaluate at multiple prefix ratios
    prefix_ratios: List[float]  # e.g. [0.1, 0.2, ..., 1.0]
    # if True, also apply random corruption noise to observed entries (optional)
    add_noise: bool = True
    noise_std: float = 0.01


def make_prefix_mask(
    base_mask: torch.Tensor,
    tau: int,
) -> torch.Tensor:
    """
    base_mask: (B, T, V) 1 observed, 0 missing
    tau: prefix length in [1..T]
    returns combined mask M where suffix is masked out regardless of base_mask
    """
    B, T, V = base_mask.shape
    device = base_mask.device
    suffix = torch.zeros((B, T, V), device=device, dtype=base_mask.dtype)
    suffix[:, :tau, :] = 1.0
    combined = base_mask * suffix
    return combined


def apply_noise_to_observed(x: torch.Tensor, mask: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return x
    noise = torch.randn_like(x) * noise_std
    return x + noise * mask


# =========================
# LEC GAN style Generator and Discriminator
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiLSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        out, _ = self.lstm(x)
        return out  # (B, T, 2H)


class GeneratorG(nn.Module):
    """
    Input:
      x_noised: (B, T, V)
      mask: (B, T, V) 1 observed, 0 missing
      s: (B, U) static
    Output:
      x_gen: (B, T, V)
    """
    def __init__(
        self,
        V: int,
        U: int,
        s_emb: int = 32,
        lstm_hidden: int = 64,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.s_embed = MLP(U, s_emb, hidden=mlp_hidden, dropout=dropout)
        # Input fusion includes x_noised, mask as channel, and static embedding broadcast
        self.in_dim = V + V + s_emb
        self.enc = BiLSTMEncoder(self.in_dim, hidden=lstm_hidden, num_layers=1, dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(2 * lstm_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, V),
        )

    def forward(self, x_noised: torch.Tensor, mask: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        B, T, V = x_noised.shape
        s_e = self.s_embed(s)  # (B, s_emb)
        s_rep = s_e.unsqueeze(1).expand(B, T, s_e.size(-1))
        inp = torch.cat([x_noised, mask, s_rep], dim=-1)
        h = self.enc(inp)
        x_gen = self.out(h)
        return x_gen


class DiscriminatorD(nn.Module):
    """
    Input:
      x: (B, T, V) imputed or real
      s: (B, U) static
      ind: (B, T, 1) temporal indicator, e.g. normalized time
    Output:
      adv_logit: (B, 1) real or fake
      cls_logit: (B, K) class
      feat: (B, F) feature vector for feature matching
    """
    def __init__(
        self,
        V: int,
        U: int,
        K: int,
        s_emb: int = 32,
        ind_dim: int = 1,
        lstm_hidden: int = 64,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.s_embed = MLP(U, s_emb, hidden=mlp_hidden, dropout=dropout)
        self.in_dim = V + s_emb + ind_dim
        self.enc = BiLSTMEncoder(self.in_dim, hidden=lstm_hidden, num_layers=1, dropout=dropout)

        # use pooled feature as global representation
        feat_dim = 2 * lstm_hidden
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(),
        )
        self.adv_head = nn.Linear(mlp_hidden, 1)
        self.cls_head = nn.Linear(mlp_hidden, K)

    def forward(self, x: torch.Tensor, s: torch.Tensor, ind: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, V = x.shape
        s_e = self.s_embed(s)
        s_rep = s_e.unsqueeze(1).expand(B, T, s_e.size(-1))
        inp = torch.cat([x, s_rep, ind], dim=-1)
        h = self.enc(inp)  # (B, T, 2H)
        # pool by mean
        pooled = h.mean(dim=1)  # (B, 2H)
        feat = self.feat_proj(pooled)  # (B, mlp_hidden)
        adv_logit = self.adv_head(feat)
        cls_logit = self.cls_head(feat)
        return adv_logit, cls_logit, feat


# =========================
# LEC GAN wrapper
# =========================
@dataclass
class LECGANConfig:
    T: int
    V: int
    U: int
    K: int
    phi_rec: float = 10.0   # weight for reconstruction
    delta_adv: float = 0.5  # mix for D: adv vs CE
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    eps: float = 1e-12


class LECGAN(nn.Module):
    def __init__(self, cfg: LECGANConfig):
        super().__init__()
        self.cfg = cfg
        self.G = GeneratorG(V=cfg.V, U=cfg.U)
        self.D = DiscriminatorD(V=cfg.V, U=cfg.U, K=cfg.K)

    def temporal_indicator(self, B: int, device: torch.device) -> torch.Tensor:
        # normalized time 0..1, shape (B, T, 1)
        t = torch.linspace(0.0, 1.0, steps=self.cfg.T, device=device).view(1, self.cfg.T, 1)
        return t.expand(B, self.cfg.T, 1)

    def impute(self, x: torch.Tensor, mask: torch.Tensor, s: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        """
        x: (B, T, V)
        mask: (B, T, V)
        return x_hat = mask*x + (1-mask)*x_gen
        """
        x_noised = x
        if noise_std > 0:
            x_noised = apply_noise_to_observed(x, mask, noise_std=noise_std)
        x_gen = self.G(x_noised, mask, s)
        x_hat = mask * x + (1.0 - mask) * x_gen
        return x_hat

    def forward_classifier(self, x_hat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        classification comes from D cls head
        returns probs: (B, K)
        """
        B = x_hat.size(0)
        ind = self.temporal_indicator(B, x_hat.device)
        _, cls_logit, _ = self.D(x_hat, s, ind)
        return F.softmax(cls_logit, dim=-1)

    def losses(
        self,
        x_real: torch.Tensor,
        mask: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
        noise_std: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns losses for G and D.
        Paper style:
          G: feature matching + phi * reconstruction
          D: delta * adv + (1-delta) * CE (CE on real only)
        """
        B = x_real.size(0)
        device = x_real.device
        ind = self.temporal_indicator(B, device)

        # generate and impute
        x_noised = x_real
        if noise_std > 0:
            x_noised = apply_noise_to_observed(x_real, mask, noise_std=noise_std)
        x_gen = self.G(x_noised, mask, s)
        x_hat = mask * x_real + (1.0 - mask) * x_gen

        # D on real and fake
        adv_real, cls_real, feat_real = self.D(x_real, s, ind)
        adv_fake, cls_fake, feat_fake = self.D(x_hat.detach(), s, ind)

        # adversarial loss for D
        # BCE with logits: real=1, fake=0
        d_adv = F.binary_cross_entropy_with_logits(adv_real, torch.ones_like(adv_real)) + \
                F.binary_cross_entropy_with_logits(adv_fake, torch.zeros_like(adv_fake))
        # classification on real only
        d_ce = F.cross_entropy(cls_real, y)

        d_loss = self.cfg.delta_adv * d_adv + (1.0 - self.cfg.delta_adv) * d_ce

        # G loss
        # feature matching between D features on real and on generated, using current D
        adv_fake_g, cls_fake_g, feat_fake_g = self.D(x_hat, s, ind)
        g_fm = F.l1_loss(feat_fake_g, feat_real.detach())
        g_rec = masked_mse(x_hat, x_real, mask, eps=self.cfg.eps)
        g_loss = g_fm + self.cfg.phi_rec * g_rec

        return {
            "x_hat": x_hat,
            "d_loss": d_loss,
            "d_adv": d_adv.detach(),
            "d_ce": d_ce.detach(),
            "g_loss": g_loss,
            "g_fm": g_fm.detach(),
            "g_rec": g_rec.detach(),
        }


# =========================
# Halting configuration
# =========================
@dataclass
class HaltingConfig:
    """
    early stopping criteria on prefix
    """
    T: int
    prefix_ratios: List[float]

    use_maxprob: bool = True
    p_thresh: float = 0.90

    use_entropy: bool = False
    e_thresh: float = 0.50

    use_stability: bool = True
    stability_runs: int = 5
    # stability threshold on mean KL or variance proxy
    kl_thresh: float = 0.02

    # imputation noise for multiple runs
    imp_noise_std: float = 0.01

    min_ratio: float = 0.0
    eps: float = 1e-12


def mean_kl_to_mean(probs_runs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    probs_runs: (R, B, K)
    return per sample mean KL(probs_r || probs_mean): (B,)
    """
    probs_runs = probs_runs.clamp_min(eps)
    p_mean = probs_runs.mean(dim=0).clamp_min(eps)  # (B, K)
    kl = (probs_runs * (probs_runs.log() - p_mean.log())).sum(dim=-1)  # (R, B)
    return kl.mean(dim=0)  # (B,)


@torch.no_grad()
def early_predict_with_halting(
    model: LECGAN,
    x_full: torch.Tensor,
    base_mask: torch.Tensor,
    s: torch.Tensor,
    hcfg: HaltingConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      pred: (B,)
      tau: (B,) in steps
    """
    device = x_full.device
    B, T, V = x_full.shape
    assert T == hcfg.T

    stopped = torch.zeros(B, dtype=torch.bool, device=device)
    tau = torch.full((B,), T, dtype=torch.long, device=device)
    final_probs = torch.zeros(B, model.cfg.K, device=device)

    for r in hcfg.prefix_ratios:
        if r < hcfg.min_ratio:
            continue
        t = int(round(r * T))
        t = max(1, min(T, t))

        mask = make_prefix_mask(base_mask, tau=t)

        if hcfg.use_stability:
            R = max(1, hcfg.stability_runs)
            probs_runs = []
            for _ in range(R):
                x_hat = model.impute(x_full, mask, s, noise_std=hcfg.imp_noise_std)
                probs = model.forward_classifier(x_hat, s)
                probs_runs.append(probs)
            probs_runs = torch.stack(probs_runs, dim=0)  # (R, B, K)
            probs_mean = probs_runs.mean(dim=0)          # (B, K)
            kl_mean = mean_kl_to_mean(probs_runs, eps=hcfg.eps)  # (B,)
        else:
            x_hat = model.impute(x_full, mask, s, noise_std=0.0)
            probs_mean = model.forward_classifier(x_hat, s)
            kl_mean = torch.zeros(B, device=device)

        pred = probs_mean.argmax(dim=-1)

        cond = torch.zeros(B, dtype=torch.bool, device=device)

        if hcfg.use_maxprob:
            maxprob = probs_mean.max(dim=-1).values
            cond = cond | (maxprob >= hcfg.p_thresh)

        if hcfg.use_entropy:
            ent = entropy_from_probs(probs_mean, eps=hcfg.eps)
            cond = cond | (ent <= hcfg.e_thresh)

        if hcfg.use_stability:
            cond = cond | (kl_mean <= hcfg.kl_thresh)

        new_stop = (~stopped) & cond
        if new_stop.any():
            tau[new_stop] = t
            stopped[new_stop] = True
            final_probs[new_stop] = probs_mean[new_stop]

        if stopped.all():
            break

    # fallback for those not stopped, take full length once
    if (~stopped).any():
        mask_full = make_prefix_mask(base_mask, tau=T)
        x_hat = model.impute(x_full, mask_full, s, noise_std=0.0)
        probs_full = model.forward_classifier(x_hat, s)
        final_probs[~stopped] = probs_full[~stopped]

    final_pred = final_probs.argmax(dim=-1)
    return final_pred, tau


# =========================
# Training and evaluation loops
# =========================
def train_one_epoch(
    model: LECGAN,
    loader,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    device: torch.device,
    mask_cfg: EarlyMaskConfig,
) -> Dict[str, float]:
    model.train()
    sum_g = 0.0
    sum_d = 0.0
    sum_rec = 0.0
    sum_fm = 0.0
    total = 0

    for batch in loader:
        # Expected batch: x, base_mask, s, y
        x, base_mask, s, y = batch
        x = x.to(device)
        base_mask = base_mask.to(device)
        s = s.to(device)
        y = y.to(device)

        B, T, V = x.shape

        # sample a random prefix for early simulation
        r = random.choice(mask_cfg.prefix_ratios)
        tau = int(round(r * mask_cfg.T))
        tau = max(1, min(mask_cfg.T, tau))
        mask = make_prefix_mask(base_mask, tau=tau)
        x_in = x
        if mask_cfg.add_noise:
            x_in = apply_noise_to_observed(x, mask, noise_std=mask_cfg.noise_std)

        # D step
        opt_d.zero_grad()
        out = model.losses(x_real=x, mask=mask, s=s, y=y, noise_std=mask_cfg.noise_std)
        d_loss = out["d_loss"]
        d_loss.backward()
        opt_d.step()

        # G step
        opt_g.zero_grad()
        out = model.losses(x_real=x, mask=mask, s=s, y=y, noise_std=mask_cfg.noise_std)
        g_loss = out["g_loss"]
        g_loss.backward()
        opt_g.step()

        bs = x.size(0)
        sum_g += g_loss.item() * bs
        sum_d += d_loss.item() * bs
        sum_rec += out["g_rec"].item() * bs
        sum_fm += out["g_fm"].item() * bs
        total += bs

    return {
        "g_loss": sum_g / max(1, total),
        "d_loss": sum_d / max(1, total),
        "g_rec": sum_rec / max(1, total),
        "g_fm": sum_fm / max(1, total),
    }


@torch.no_grad()
def eval_early_metrics(
    model: LECGAN,
    loader,
    device: torch.device,
    hcfg: HaltingConfig,
) -> Dict[str, float]:
    model.eval()
    all_pred = []
    all_y = []
    all_tau = []

    for batch in loader:
        x, base_mask, s, y = batch
        x = x.to(device)
        base_mask = base_mask.to(device)
        s = s.to(device)
        y = y.to(device)

        pred, tau = early_predict_with_halting(model, x, base_mask, s, hcfg)

        all_pred.append(pred.detach().cpu())
        all_y.append(y.detach().cpu())
        all_tau.append(tau.detach().cpu())

    pred = torch.cat(all_pred, dim=0)
    y = torch.cat(all_y, dim=0)
    tau = torch.cat(all_tau, dim=0).float()

    f1 = macro_f1(pred, y, n_classes=model.cfg.K)
    el = float((tau / hcfg.T).mean().item())
    hm = harmonic_mean_f1_earliness(f1, el)

    return {"macro_f1": f1, "earliness_el": el, "hm": hm}


def run_experiment(
    model: LECGAN,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    mask_cfg: EarlyMaskConfig,
    hcfg: HaltingConfig,
) -> Dict[str, float]:
    opt_g = torch.optim.Adam(model.G.parameters(), lr=model.cfg.lr_g)
    opt_d = torch.optim.Adam(model.D.parameters(), lr=model.cfg.lr_d)

    best_hm = -1.0
    best_epoch = -1
    best_state = None

    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, opt_g, opt_d, device, mask_cfg)
        val = eval_early_metrics(model, val_loader, device, hcfg)

        if val["hm"] > best_hm:
            best_hm = val["hm"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch {ep:03d} | "
            f"g {tr['g_loss']:.4f} d {tr['d_loss']:.4f} rec {tr['g_rec']:.4f} fm {tr['g_fm']:.4f} | "
            f"F1 {val['macro_f1']:.4f} EL {val['earliness_el']:.4f} HM {val['hm']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {"best_hm": best_hm, "best_epoch": best_epoch}


# =========================
# Minimal example usage
# =========================
if __name__ == "__main__":
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # You must set these to match your dataset
    T = 100
    V = 8
    U = 4
    K = 5

    cfg = LECGANConfig(T=T, V=V, U=U, K=K, phi_rec=10.0, delta_adv=0.5, lr_g=1e-3, lr_d=1e-3)
    model = LECGAN(cfg).to(device)

    # train_loader and val_loader must yield: (x, base_mask, s, y)
    # x: (B, T, V)
    # base_mask: (B, T, V) 1 observed, 0 missing
    # s: (B, U)
    # y: (B,)

    mask_cfg = EarlyMaskConfig(
        T=T,
        prefix_ratios=[0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        add_noise=True,
        noise_std=0.01,
    )

    hcfg = HaltingConfig(
        T=T,
        prefix_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        use_maxprob=True,
        p_thresh=0.90,
        use_entropy=False,
        e_thresh=0.50,
        use_stability=True,
        stability_runs=5,
        kl_thresh=0.02,
        imp_noise_std=0.01,
        min_ratio=0.1,
    )

    # Example:
    # result = run_experiment(model, train_loader, val_loader, device, epochs=30, mask_cfg=mask_cfg, hcfg=hcfg)
    # print(result)
    pass
