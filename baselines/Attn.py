import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------
def compute_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    probs: (B, S, C) or (B, C)
    return entropy: (B, S) or (B,)
    """
    probs = probs.clamp_min(eps)
    ent = -(probs * probs.log()).sum(dim=-1)
    return ent


def fft_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, L, N)
    return: (B, L, N) like representation for convenience
    We use rfft along time dimension and take magnitude.
    We then pad/truncate back to L for simplicity.
    """
    # rfft length = L//2 + 1
    Xf = torch.fft.rfft(x, dim=1)  # (B, Lf, N) complex
    mag = torch.abs(Xf)            # (B, Lf, N) real
    L = x.size(1)
    Lf = mag.size(1)

    if Lf == L:
        return mag
    if Lf < L:
        pad = torch.zeros(mag.size(0), L - Lf, mag.size(2), device=mag.device, dtype=mag.dtype)
        return torch.cat([mag, pad], dim=1)
    # Lf > L (rare)
    return mag[:, :L, :]


def unfold_segments(x: torch.Tensor, seg_len: int, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, L, N)
    returns:
      segments: (B, S, seg_len, N)
      start_idx: (S,) start indices in [0, L-seg_len]
    """
    B, L, N = x.shape
    if seg_len > L:
        raise ValueError(f"seg_len({seg_len}) must be <= L({L})")

    # unfold along time dimension
    # x_unf: (B, S, seg_len, N)
    x_unf = x.unfold(dimension=1, size=seg_len, step=stride)  # (B, S, N, seg_len)
    x_unf = x_unf.permute(0, 1, 3, 2).contiguous()  # (B, S, seg_len, N)
    S = x_unf.size(1)

    start_idx = torch.arange(0, 1 + (L - seg_len), stride, device=x.device)
    if start_idx.numel() != S:
        # Safety in case of edge behavior
        start_idx = start_idx[:S]

    return x_unf, start_idx


def pad_segment_to_L(seg: torch.Tensor, L: int) -> torch.Tensor:
    """
    seg: (B, S, seg_len, N)
    returns padded: (B, S, L, N) by zero padding at the end
    """
    B, S, seg_len, N = seg.shape
    if seg_len == L:
        return seg
    if seg_len > L:
        return seg[:, :, :L, :]
    pad_len = L - seg_len
    pad = torch.zeros(B, S, pad_len, N, device=seg.device, dtype=seg.dtype)
    return torch.cat([seg, pad], dim=2)


# -------------------------
# Backbone: MDDNN (Time and Frequency branches)
# -------------------------
class ConvLSTMBranch(nn.Module):
    """
    One branch used for time domain or frequency domain.
    Input: (B, L, N)  -> treat N as channels for Conv1d: (B, N, L)
    Two Conv1d layers + BN + ReLU + Dropout, then LSTM.
    Output: feature vector (B, H)
    """
    def __init__(
        self,
        n_vars: int,
        conv1_out: int = 64,
        conv2_out: int = 32,
        kernel_size: int = 7,
        lstm_hidden: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_vars, conv1_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=conv2_out, hidden_size=lstm_hidden, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N)
        x = x.transpose(1, 2)  # (B, N, L)
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)  # (B, L, C)
        out, _ = self.lstm(x)  # (B, L, H)
        feat = out[:, -1, :]   # last hidden over time
        return feat


class MDDNN(nn.Module):
    """
    MDDNN backbone.
    Inputs:
      x_time: (B, L, N)
      x_freq: (B, L, N)
    Output:
      logits: (B, C)
      probs:  (B, C)
    """
    def __init__(
        self,
        n_vars: int,
        n_classes: int,
        conv1_out: int = 64,
        conv2_out: int = 32,
        kernel_size: int = 7,
        lstm_hidden: int = 64,
        dropout: float = 0.5,
        fusion_hidden: int = 128,
    ):
        super().__init__()
        self.time_branch = ConvLSTMBranch(
            n_vars=n_vars,
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            kernel_size=kernel_size,
            lstm_hidden=lstm_hidden,
            dropout=dropout,
        )
        self.freq_branch = ConvLSTMBranch(
            n_vars=n_vars,
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            kernel_size=kernel_size,
            lstm_hidden=lstm_hidden,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * lstm_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, n_classes),
        )

    def forward(self, x_time: torch.Tensor, x_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ft = self.time_branch(x_time)
        ff = self.freq_branch(x_freq)
        fused = torch.cat([ft, ff], dim=-1)
        logits = self.fusion(fused)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


# -------------------------
# ETSCM Wrapper
# -------------------------
@dataclass
class ETSCMConfig:
    L: int                      # full length
    seg_ratio: float = 0.2      # segment length = int(L * seg_ratio)
    stride: int = 1
    top_k: int = 5
    eps: float = 1e-12


class ETSCM(nn.Module):
    """
    Implements ETSCM logic:
      1) Make segments with sliding window
      2) Pad segments to length L
      3) For each segment, run MDDNN
      4) Compute entropy, pick top K lowest entropy
      5) Compute weights from entropy
      6) Weighted vote for prediction
      7) Weighted loss for training
    """
    def __init__(self, backbone: MDDNN, cfg: ETSCMConfig):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

    def forward(
        self,
        x_full: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_spans: bool = True,
    ):
        """
        x_full: (B, L, N)
        y: (B,) optional
        returns dict with:
          probs_final: (B, C)
          pred: (B,)
          loss: scalar (if y provided)
          topk_start: (B, K) start indices of selected segments
          topk_entropy: (B, K)
          topk_weights: (B, K)
        """
        B, L, N = x_full.shape
        assert L == self.cfg.L, f"Expected L={self.cfg.L}, got {L}"

        seg_len = max(1, int(round(L * self.cfg.seg_ratio)))
        seg_len = min(seg_len, L)

        # 1) segments from x_full
        segs, start_idx = unfold_segments(x_full, seg_len=seg_len, stride=self.cfg.stride)
        # segs: (B, S, seg_len, N)
        B, S, seg_len, N = segs.shape

        # 2) pad segments to L for backbone
        segs_L = pad_segment_to_L(segs, L=L)  # (B, S, L, N)

        # 3) time and freq inputs for each segment
        # batch all segments for single forward
        segs_flat = segs_L.reshape(B * S, L, N)
        x_time = segs_flat
        x_freq = fft_magnitude(segs_flat)

        _, probs = self.backbone(x_time, x_freq)  # (B*S, C)
        C = probs.size(-1)

        probs = probs.reshape(B, S, C)            # (B, S, C)

        # 4) entropy for each segment
        ent = compute_entropy(probs, eps=self.cfg.eps)  # (B, S)

        # pick K lowest entropy
        K = min(self.cfg.top_k, S)
        ent_k, idx_k = torch.topk(ent, k=K, dim=1, largest=False, sorted=True)  # (B, K)

        # gather probs for selected segments
        probs_k = torch.gather(
            probs,
            dim=1,
            index=idx_k.unsqueeze(-1).expand(B, K, C)
        )  # (B, K, C)

        # 5) weights from entropy
        # E_max = log(|C|) for uniform distribution
        E_max = math.log(C)
        D = (E_max - ent_k).clamp_min(0.0)  # (B, K)
        denom = D.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps)
        w = D / denom  # (B, K)

        # 6) weighted vote, here implemented as weighted sum of probabilities
        probs_final = (probs_k * w.unsqueeze(-1)).sum(dim=1)  # (B, C)
        pred = probs_final.argmax(dim=-1)                     # (B,)

        out = {
            "probs_final": probs_final,
            "pred": pred,
            "topk_entropy": ent_k,
            "topk_weights": w,
        }

        if return_spans:
            # Map idx_k to start indices
            # start_idx: (S,)
            # topk_start: (B, K)
            topk_start = start_idx[idx_k]  # indexing a 1D tensor with (B, K)
            out["topk_start"] = topk_start
            out["seg_len"] = seg_len

        # 7) weighted loss on selected segments
        if y is not None:
            # segment loss: CE for each selected segment
            # use log probs safely
            logp_k = (probs_k.clamp_min(self.cfg.eps)).log()  # (B, K, C)
            # pick log prob of true class
            y_expand = y.view(B, 1, 1).expand(B, K, 1)
            logp_y = torch.gather(logp_k, dim=2, index=y_expand).squeeze(-1)  # (B, K)
            loss_k = -logp_y                                                # (B, K)
            loss = (loss_k * w).sum(dim=1).mean()
            out["loss"] = loss

        return out


# -------------------------
# Example training loop
# -------------------------
def train_one_epoch(
    model: ETSCM,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x, y=y, return_spans=False)
        loss = out["loss"]
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total += bs

    return total_loss / max(1, total)


@torch.no_grad()
def eval_epoch(model: ETSCM, loader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x, y=y, return_spans=False)
        pred = out["pred"]
        loss = out["loss"]

        correct += (pred == y).sum().item()
        total += x.size(0)
        total_loss += loss.item() * x.size(0)

    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return acc, avg_loss


# -------------------------
# Minimal usage example (pseudo)
# -------------------------
if __name__ == "__main__":
    # Suppose:
    # L = 300, Nvars = 20, n_classes = 10
    L = 300
    Nvars = 20
    C = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = MDDNN(
        n_vars=Nvars,
        n_classes=C,
        conv1_out=64,
        conv2_out=32,
        kernel_size=max(3, int(0.1 * L) | 1),  # odd kernel roughly 10% of L
        lstm_hidden=64,
        dropout=0.5,
        fusion_hidden=128,
    )

    cfg = ETSCMConfig(L=L, seg_ratio=0.2, stride=1, top_k=5)
    model = ETSCM(backbone=backbone, cfg=cfg).to(device)

    # You need to provide DataLoader that yields (x, y)
    # x: (B, L, Nvars), y: (B,)
    # train_loader, val_loader = ...

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # for epoch in range(epochs):
    #     tr_loss = train_one_epoch(model, train_loader, optimizer, device)
    #     val_acc, val_loss = eval_epoch(model, val_loader, device)
    #     print(epoch, tr_loss, val_acc, val_loss)

    # Inference with interpretability spans
    # x_test: (B, L, Nvars)
    # out = model(x_test.to(device), y=None, return_spans=True)
    # out["topk_start"], out["seg_len"] gives highlighted segments
    pass


import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# -------------------------
# Macro-F1 (no sklearn)
# -------------------------
@torch.no_grad()
def macro_f1_from_preds(pred: torch.Tensor, y: torch.Tensor, n_classes: int, eps: float = 1e-12) -> float:
    """
    pred: (N,) int64
    y:    (N,) int64
    """
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
    Higher is better:
      f1 in [0,1]
      el in [0,1] (lower is better)
    Use (1-el) as timeliness score.
    """
    t = 1.0 - el
    return float((2.0 * f1 * t) / (f1 + t + eps))


# -------------------------
# Early inference using halting
# -------------------------
@dataclass
class EarlyInferConfig:
    L: int
    prefix_ratios: List[float]            # 예: [0.1, 0.15, 0.2, ... , 1.0]
    min_ratio: float = 0.0               # 너무 이른 구간에서 멈추지 않게 하려면 >0
    use_maxprob: bool = True
    p_thresh: float = 0.9                # max prob threshold
    use_entropy: bool = False
    e_thresh: float = 0.5                # entropy threshold
    eps: float = 1e-12


@torch.no_grad()
def early_predict_batch(
    model,
    x_full: torch.Tensor,
    cfg: EarlyInferConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    model: ETSCM
    x_full: (B, L, Nvars) full sequence (already padded if needed)
    returns:
      pred: (B,)
      tau:  (B,) stopping time in steps (1..L)
      probs_final: (B, C) final probs at stopping time
    """
    device = x_full.device
    B, L, _ = x_full.shape
    assert L == cfg.L

    # init
    stopped = torch.zeros(B, dtype=torch.bool, device=device)
    tau = torch.full((B,), L, dtype=torch.long, device=device)  # default stop at end
    final_probs = None
    final_pred = None

    # iterate prefixes
    for r in cfg.prefix_ratios:
        r = float(r)
        if r < cfg.min_ratio:
            continue

        t = int(round(r * L))
        t = max(1, min(L, t))

        # build prefix-padded input
        x_pref = torch.zeros_like(x_full)
        x_pref[:, :t, :] = x_full[:, :t, :]

        out = model(x_pref, y=None, return_spans=False)
        probs = out["probs_final"]  # (B, C)
        pred = probs.argmax(dim=-1)

        # halting score
        cond = torch.zeros(B, dtype=torch.bool, device=device)
        if cfg.use_maxprob:
            maxprob = probs.max(dim=-1).values
            cond = cond | (maxprob >= cfg.p_thresh)
        if cfg.use_entropy:
            ent = -(probs.clamp_min(cfg.eps) * probs.clamp_min(cfg.eps).log()).sum(dim=-1)
            cond = cond | (ent <= cfg.e_thresh)

        # stop newly satisfied samples
        new_stop = (~stopped) & cond
        if new_stop.any():
            tau[new_stop] = t
            stopped[new_stop] = True

            # store final outputs for those samples
            if final_probs is None:
                final_probs = probs.clone()
                final_pred = pred.clone()
            else:
                final_probs[new_stop] = probs[new_stop]
                final_pred[new_stop] = pred[new_stop]

        # if all stopped, break
        if stopped.all():
            break

    # for samples never stopped early, do one final full pass to get consistent probs
    if (~stopped).any():
        out = model(x_full, y=None, return_spans=False)
        probs = out["probs_final"]
        pred = probs.argmax(dim=-1)

        if final_probs is None:
            final_probs = probs
            final_pred = pred
        else:
            final_probs[~stopped] = probs[~stopped]
            final_pred[~stopped] = pred[~stopped]

    return final_pred, tau, final_probs


# -------------------------
# Eval loop: Macro-F1, EL, HM
# -------------------------
@torch.no_grad()
def eval_early_metrics(
    model,
    loader,
    n_classes: int,
    cfg: EarlyInferConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    loader yields (x, y)
      x: (B, L, Nvars)
      y: (B,)
    """
    model.eval()

    all_pred = []
    all_y = []
    all_tau = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred, tau, _ = early_predict_batch(model, x, cfg)

        all_pred.append(pred.detach().cpu())
        all_y.append(y.detach().cpu())
        all_tau.append(tau.detach().cpu())

    pred = torch.cat(all_pred, dim=0)
    y = torch.cat(all_y, dim=0)
    tau = torch.cat(all_tau, dim=0).float()

    f1 = macro_f1_from_preds(pred, y, n_classes=n_classes)
    el = float((tau / cfg.L).mean().item())
    hm = harmonic_mean_f1_earliness(f1, el)

    return {"macro_f1": f1, "earliness_el": el, "hm": hm}


# -------------------------
# Full experiment template
# -------------------------
def run_experiment(
    model,
    train_loader,
    val_loader,
    n_classes: int,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    early_cfg: Optional[EarlyInferConfig] = None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if early_cfg is None:
        # 기본 prefix grid
        prefix_ratios = [i / 20 for i in range(1, 21)]  # 0.05 .. 1.0
        early_cfg = EarlyInferConfig(
            L=model.cfg.L,
            prefix_ratios=prefix_ratios,
            min_ratio=0.1,
            use_maxprob=True,
            p_thresh=0.9,
            use_entropy=False,
            e_thresh=0.5,
        )

    best_hm = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # train (ETSCM weighted loss)
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x, y=y, return_spans=False)
            loss = out["loss"]
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs

        train_loss = total_loss / max(1, total)

        # eval early metrics on val
        metrics = eval_early_metrics(model, val_loader, n_classes, early_cfg, device)

        # model selection by HM
        if metrics["hm"] > best_hm:
            best_hm = metrics["hm"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} | "
            f"F1 {metrics['macro_f1']:.4f} | EL {metrics['earliness_el']:.4f} | HM {metrics['hm']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {"best_hm": best_hm}