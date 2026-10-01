#!/usr/bin/env python3
"""Extract and whiten background windows from background_triggers.csv, each paired with real witness
and other-detector data for the same GPS window -> auto_data/<DET>/whitened_background_full.h5."""

from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import common
import config as cfg

max_glitch_duration = 8.0
n_samples = None
random_seed = 42

candidates = pd.read_csv(cfg.background_triggers_csv)["gps_start"].to_numpy()
print(f"{cfg.detector}: {len(candidates)} candidate windows")

# Drop candidates whose PSD + kernel footprint touches a long strain glitch.
trig = pd.read_csv(common.find_trigger_csv("GDS-CALIB_STRAIN"))
long_trig = trig[trig["tend"] - trig["tstart"] > max_glitch_duration]
fp_start = candidates - cfg.whiten_psd_length - cfg.pad_seconds
fp_end = candidates + cfg.window + cfg.pad_seconds
near_long = ((fp_start[:, None] < long_trig["tend"].to_numpy()[None, :])
             & (fp_end[:, None] > long_trig["tstart"].to_numpy()[None, :])).any(axis=1)
candidates = candidates[~near_long]
print(f"dropped near a glitch > {max_glitch_duration}s: {int(near_long.sum())}")

np.random.default_rng(random_seed).shuffle(candidates)
if n_samples is not None:
    candidates = candidates[:n_samples]
print(f"selected for extraction: {len(candidates)}")

manifests = common.load_manifests()
transforms = common.build_transforms(common.manifest_rates(manifests))
seq_len = int(cfg.window * cfg.target_sample_rate)

strain_out, other_out, witness_out, gps_out = [], [], [], []
drop_reasons = Counter()
cache = {}

for win_start in tqdm(candidates, desc=f"{cfg.detector}+{cfg.other_detector}"):
    seg_start = win_start - cfg.whiten_psd_length - cfg.pad_seconds
    seg_end = win_start + cfg.window + cfg.pad_seconds

    strain, reason = common.extract_channel(manifests["strain"], cache, cfg.detector, None,
                                            seg_start, seg_end, transforms, seq_len)
    if strain is None:
        drop_reasons[f"strain:{reason}"] += 1
        continue
    other_strain, witness, reason = common.extract_companions(manifests, cache, seg_start, seg_end, transforms, seq_len)
    if other_strain is None:
        drop_reasons[reason] += 1
        continue

    strain_out.append(strain)
    other_out.append(other_strain)
    witness_out.append(witness)
    gps_out.append(win_start + cfg.window / 2)

common.close_all(cache)

print(f"\nValid samples: {len(gps_out)} / {len(candidates)}")
for reason, count in drop_reasons.most_common():
    print(f"  {reason}: {count}")

common.save_h5(cfg.background_h5, strain_out, other_out, witness_out, gps_out)
