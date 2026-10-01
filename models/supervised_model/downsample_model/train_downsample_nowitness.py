"""Kept only to load old checkpoints; new runs use DownsampleModule(no_witness=True)."""

from train_downsample import DownsampleModule


class NoWitnessDownsampleModule(DownsampleModule):
    def __init__(self, *args, **kwargs):
        kwargs["no_witness"] = True
        super().__init__(*args, **kwargs)
