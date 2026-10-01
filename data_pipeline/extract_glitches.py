#!/usr/bin/env python3
"""Extract and whiten witness-coincident glitches (low/high frequency split, jittered windows), each paired
with real witness and other-detector data -> time_data/<DET>/whitened_glitches.h5 and whitened_high_glitches.h5."""

import pandas as pd

import common
import config as cfg

max_low_freq_samples = 10000
min_snr = 7
nyquist = cfg.target_sample_rate / 2.0


glitches = pd.read_csv(cfg.coincidence_csv)
glitches = glitches[glitches["snr"] >= min_snr].copy()
low = glitches["frequency"] <= nyquist
glitches = pd.concat([
    glitches[low].sort_values("witness_match_count", ascending=False).head(max_low_freq_samples),
    glitches[~low].sort_values("witness_match_count", ascending=False),
])
print(f"{cfg.detector}: {len(glitches)} glitches with SNR >= {min_snr}")

common.extract_glitch_bands(glitches, cfg.glitch_h5, cfg.glitch_high_h5, max_low=max_low_freq_samples)
