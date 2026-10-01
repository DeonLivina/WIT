"""Lightning module for the DownsampleAuxModel (same loss and training step as SupervisedModule)."""

from downsample_model import DownsampleAuxModel
from train import SupervisedModule


class DownsampleModule(SupervisedModule):
    def __init__(self, n_witness, n_classes, d_model=32, mamba_layers=4, proj_dim=8, lr=1e-3,
                 temperature=0.1, supcon_weight=0.05, witness_label="detector", witness_supcon_weight=0.05,
                 no_witness=False):
        super().__init__(n_witness, n_classes, d_model, mamba_layers, proj_dim, lr, temperature,
                         supcon_weight, witness_label, witness_supcon_weight, no_witness)
        self.model = DownsampleAuxModel(n_witness, n_classes, d_model, mamba_layers, proj_dim)
