"""
LEC-GAN placeholder - Learning to Early Classify with GAN.
Full implementation requires external LEC-GAN code.
This module provides a minimal stub that uses a simple early classifier.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEarlyClassifier(nn.Module):
    """Minimal early classifier: LSTM + halting at prefix."""

    def __init__(self, ninp, nclasses, nhid=64):
        super().__init__()
        self.lstm = nn.LSTM(ninp, nhid, batch_first=True)
        self.fc = nn.Linear(nhid, nclasses)

    def forward(self, x, prefix_ratio=0.5):
        """x: (B, T, C). Use first prefix_ratio of sequence."""
        B, T, C = x.shape
        t = max(1, int(T * prefix_ratio))
        out, _ = self.lstm(x[:, :t, :])
        logits = self.fc(out[:, -1, :])
        return logits


def create_lecgan_model(ninp, nclasses, nhid=64):
    return SimpleEarlyClassifier(ninp, nclasses, nhid)
