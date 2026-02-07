"""
ETSCM-style segment-based early classification with entropy-based segment selection.
Simple backbone (MLP) for segment-level features, unfold segments, pad, classify.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def unfold_segments(x, segment_len, stride):
    """x: (B, T, C). Returns (B, num_segments, segment_len, C)."""
    B, T, C = x.shape
    segments = []
    for start in range(0, T - segment_len + 1, stride):
        seg = x[:, start : start + segment_len, :]
        segments.append(seg)
    if not segments:
        seg = x[:, -segment_len:, :]
        segments = [seg]
    return torch.stack(segments, dim=1)


def pad_segment_to_L(seg, L):
    """seg: (B, S, C). Pad or truncate to length L. Returns (B, L, C)."""
    B, S, C = seg.shape
    if S >= L:
        return seg[:, :L, :]
    pad_size = L - S
    return F.pad(seg, (0, 0, 0, pad_size), mode="constant", value=0)


class MDDNNBackbone(nn.Module):
    """Simple MLP backbone for segment features."""

    def __init__(self, input_dim, hidden_dim=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_out(h)


class AttnEarlyClassifier(nn.Module):
    """
    Segment-based early classifier.
    - Unfold input into segments
    - Pad segments to fixed length
    - Backbone produces logits per segment
    - Select segment by minimum entropy (or by num_prefix_steps)
    """

    def __init__(self, ninp, nclasses, segment_len=16, stride=8, hidden_dim=64):
        super().__init__()
        self.ninp = ninp
        self.nclasses = nclasses
        self.segment_len = segment_len
        self.stride = stride
        self.L = segment_len
        self.backbone = MDDNNBackbone(ninp * self.L, hidden_dim, nclasses)

    def forward(self, x, num_prefix_steps=None, test=False):
        """
        x: (B, T, C)
        num_prefix_steps: if set, use first N segments for prediction (early inference)
        Returns: logits (B, nclasses), segment_idx (B,) or tau (earliness)
        """
        B, T, C = x.shape
        segs = unfold_segments(x, self.segment_len, self.stride)
        S = segs.shape[1]
        segs_flat = segs.reshape(B, S, -1)
        logits_all = self.backbone(segs_flat)
        probs_all = F.softmax(logits_all, dim=-1)
        entropy_all = -(probs_all * (probs_all + 1e-8).log()).sum(dim=-1)

        if num_prefix_steps is not None:
            k = min(num_prefix_steps, S)
            seg_range = list(range(k))
            idx = torch.tensor(seg_range, device=x.device).unsqueeze(0).expand(B, -1)
            best_idx = entropy_all[:, :k].argmin(dim=1)
            logits = logits_all[torch.arange(B, device=x.device), best_idx]
            tau = (best_idx.float() + 1) / max(S, 1)
        else:
            best_idx = entropy_all.argmin(dim=1)
            logits = logits_all[torch.arange(B, device=x.device), best_idx]
            tau = (best_idx.float() + 1) / max(S, 1)

        return logits, tau


def create_attn_model(ninp, nclasses, segment_len=16, stride=8, hidden_dim=64):
    return AttnEarlyClassifier(ninp, nclasses, segment_len, stride, hidden_dim)
