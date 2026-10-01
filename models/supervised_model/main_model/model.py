"""AuxModel (pos_abi ablation winner): separate strain / witness encoders -> Mamba fusion trunk -> heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaStack(nn.Module):
    def __init__(self, d_model, layers):
        super().__init__()
        self.blocks = nn.ModuleList([Mamba(d_model=d_model) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])

    def forward(self, x):
        for block, norm in zip(self.blocks, self.norms):
            x = x + block(norm(x))
        return x


class StreamEncoder(nn.Module):
    """Conv stem (stride 1, no downsampling) + Mamba stack. (B, T, C) -> (B, T, d)."""

    def __init__(self, in_ch, d_model, layers):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, d_model, 7, padding=3), nn.GELU(),
            nn.Conv1d(d_model, d_model, 5, padding=2), nn.GELU(),
        )
        self.mamba = MambaStack(d_model, layers)

    def forward(self, x):
        return self.mamba(self.stem(x.transpose(1, 2)).transpose(1, 2))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class AuxModel(nn.Module):
    def __init__(self, n_witness, n_classes, d_model=32, mamba_layers=4, proj_dim=8):
        super().__init__()
        self.strain_encoder = StreamEncoder(1, d_model, mamba_layers)
        self.witness_encoder = StreamEncoder(n_witness, d_model, mamba_layers)
        self.fusion_proj = nn.Linear(2 * d_model, d_model)
        self.trunk = MambaStack(d_model, mamba_layers)
        self.strain_proj = ProjectionHead(d_model, proj_dim)
        self.witness_proj = ProjectionHead(d_model, proj_dim)
        self.classifier = nn.Linear(proj_dim, n_classes)

    def forward(self, x):
        """x: (B, T, 1 + n_witness), strain first."""
        s = self.strain_encoder(x[..., :1])
        w = self.witness_encoder(x[..., 1:])
        h = self.trunk(self.fusion_proj(torch.cat([s, w], dim=-1))).max(dim=1).values
        strain_z = self.strain_proj(h)
        return {"strain_z": strain_z, "witness_z": self.witness_proj(h), "logits": self.classifier(strain_z)}
