#!/usr/bin/env python3
"""Build the "signal" class: inject CBC waveforms into real background strain (SNRs bootstrapped from
the low-frequency glitch pool) with real witness + other-detector data """

import config as cfg
from inject_common import yaml_cfg, device, rate, run_injection
from injections import _time_jitter
from waveforms import generate_signals

n_target = 2000
candidate_seed = 43
snr_bootstrap_seed = 7


def make_waveform(sig_cfg, raw_inj):
    waveform, _ = generate_signals(sig_cfg, device)
    waveform, _ = _time_jitter(waveform.double(), yaml_cfg, rate, device)
    return waveform


if __name__ == "__main__":
    run_injection(make_waveform, cfg.signal_h5, n_target, candidate_seed, snr_bootstrap_seed, "Injecting signals")
