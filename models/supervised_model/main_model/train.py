"""Lightning module for the AuxModel.

    loss = CE(class) + supcon_weight * SupCon(strain_z, class) + witness_supcon_weight * SupCon(witness_z, detector)
"""

import lightning as L
import torch
import torch.nn.functional as F
from losses import SupervisedSimCLRLoss

from model import AuxModel


class SupervisedModule(L.LightningModule):
    def __init__(self, n_witness, n_classes, d_model=32, mamba_layers=4, proj_dim=8, lr=1e-3,
                 temperature=0.1, supcon_weight=0.05, witness_label="detector", witness_supcon_weight=0.05,
                 no_witness=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = AuxModel(n_witness, n_classes, d_model, mamba_layers, proj_dim)
        self.supcon = SupervisedSimCLRLoss(temperature)

    def forward(self, x):
        if self.hparams.no_witness:
            x = torch.cat([x[..., :1], torch.zeros_like(x[..., 1:])], dim=-1)
        return self.model(x)

    def _step(self, batch, stage):
        x, y, y_det = batch[0], batch[1], batch[2]
        out = self(x)
        w_labels = y_det if self.hparams.witness_label == "detector" else y

        ce = F.cross_entropy(out["logits"], y)
        strain_con = self.supcon(out["strain_z"].unsqueeze(1), y)
        witness_con = self.supcon(out["witness_z"].unsqueeze(1), w_labels)
        loss = ce + self.hparams.supcon_weight * strain_con + self.hparams.witness_supcon_weight * witness_con

        self.log(f"{stage}_ce", ce)
        self.log(f"{stage}_strain_supcon", strain_con)
        self.log(f"{stage}_witness_supcon", witness_con)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", (out["logits"].argmax(-1) == y).float().mean(), prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
