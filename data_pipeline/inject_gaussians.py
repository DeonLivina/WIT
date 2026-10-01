#!/usr/bin/env python3
"""Build the strain-only "blip" class: inject SineGaussian blips into real background strain only."""

import config as cfg
from inject_common import device, f_max, f_min, rate, run_injection
from injections import _time_jitter
from waveforms import generate_glitch_sources
from witness import _rms_normalize, bandlimit

n_target = 2000
candidate_seed = 46
snr_bootstrap_seed = 47


def make_waveform(sig_cfg, raw_inj):
    blip, _ = generate_glitch_sources(sig_cfg, device)
    blip = _rms_normalize(bandlimit(blip, rate, f_min, f_max)).unsqueeze(1)
    blip, _ = _time_jitter(blip, sig_cfg, rate, device)
    return blip * raw_inj.std(dim=-1, keepdim=True)


if __name__ == "__main__":
    run_injection(make_waveform, cfg.blip_h5, n_target, candidate_seed, snr_bootstrap_seed,
                  "Injecting strain-only blips")
