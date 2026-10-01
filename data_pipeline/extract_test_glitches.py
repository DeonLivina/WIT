#!/usr/bin/env python3
"""Extract and whiten every SNR >= 7 strain Omicron trigger (<= 1 s long) in the test period, with or without a
witness coincidence, so the set represents the whole glitch population. Same windows and companions as
extract_glitches.py """

import pandas as pd

import common
import config as cfg

max_low_freq_samples = 10000
min_snr = 7
sample_seed = 45
nyquist = cfg.target_sample_rate / 2.0


glitches = pd.read_csv(common.find_trigger_csv("GDS-CALIB_STRAIN"))
duration = glitches["tend"] - glitches["tstart"]
glitches = glitches[(glitches["snr"] >= min_snr) & (glitches["tstart"] >= cfg.test_cutoff_gps)
                    & (duration <= cfg.window)].copy()
print(f"{cfg.detector}: {len(glitches)} strain triggers with SNR >= {min_snr} from {cfg.test_cutoff_utc} UTC")

# Random (not ranked) subsample if there are too many low-frequency ones, to keep the population unbiased.
low = glitches["frequency"] <= nyquist
n_low = int(low.sum())
if n_low > max_low_freq_samples:
    print(f"Randomly keeping {max_low_freq_samples} of {n_low} low-frequency triggers")
glitches = pd.concat([
    glitches[low].sample(n=min(n_low, max_low_freq_samples), random_state=sample_seed),
    glitches[~low],
])

common.extract_glitch_bands(glitches, cfg.test_glitch_h5, cfg.test_glitch_high_h5, max_low=max_low_freq_samples)
