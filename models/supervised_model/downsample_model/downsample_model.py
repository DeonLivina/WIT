"""Downsample variant of the AuxModel: conv downsamplers (4096 -> 1024) replace the per-stream Mamba encoders."""

import torch
import torch.nn as nn

from model import MambaStack, ProjectionHead


class Downsampler(nn.Module):
    """2 x Conv1d(kernel 16, stride 2, padding 7): T -> T/4 (4096 -> 1024). (B, C, T) -> (B, d, T/4)."""

    def __init__(self, in_ch, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model, 16, stride=2, padding=7), nn.GELU(),
            nn.Conv1d(d_model, d_model, 16, stride=2, padding=7), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DownsampleAuxModel(nn.Module):
    """Conv downsamplers instead of the per-stream Mamba encoders; Mamba only in the fusion trunk.

    strain  (B, T, 1)  -> Downsampler                         -> (B, T/4, d)
    witness (B, T, W)  -> fold to (B*W, 1, T) -> shared Downsampler -> unfold (B, T/4, W*d)
    concat (B, T/4, (1+W)*d) -> Linear -> d -> Mamba fusion -> max pool -> strain_z, witness_z
    """

    def __init__(self, n_witness, n_classes, d_model=32, mamba_layers=4, proj_dim=8):
        super().__init__()
        self.strain_down = Downsampler(1, d_model)
        self.witness_down = Downsampler(1, d_model)
        self.fusion_proj = nn.Linear((1 + n_witness) * d_model, d_model)
        self.trunk = MambaStack(d_model, mamba_layers)
        self.strain_proj = ProjectionHead(d_model, proj_dim)
        self.witness_proj = ProjectionHead(d_model, proj_dim)
        self.classifier = nn.Linear(proj_dim, n_classes)

    def forward(self, x):
        """x: (B, T, 1 + n_witness), strain first."""
        B, T, C = x.shape
        s = self.strain_down(x[..., :1].transpose(1, 2)).transpose(1, 2)            # (B, T/4, d)
        w = self.witness_down(x[..., 1:].transpose(1, 2).reshape(B * (C - 1), 1, T))  # (B*W, d, T/4)
        w = w.reshape(B, C - 1, -1, w.shape[-1]).permute(0, 3, 1, 2).flatten(2)      # (B, T/4, W*d)
        h = self.trunk(self.fusion_proj(torch.cat([s, w], dim=-1))).max(dim=1).values
        strain_z = self.strain_proj(h)
        return {"strain_z": strain_z, "witness_z": self.witness_proj(h), "logits": self.classifier(strain_z)}
