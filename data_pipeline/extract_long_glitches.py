#!/usr/bin/env python3
"""Cut low-frequency (<= 2048 Hz) glitches longer than 1 s into consecutive 1 s windows (15.5 s -> 16),
whitened and paired with real witness + other-detector data."""

import numpy as np
import pandas as pd
from tqdm import tqdm

import common
import config as cfg

min_snr = 7
nyquist = cfg.target_sample_rate / 2.0

glitches = pd.read_csv(cfg.coincidence_csv)
duration = glitches["tend"] - glitches["tstart"]
glitches = glitches[(glitches["snr"] >= min_snr) & (glitches["frequency"] <= nyquist)
                    & (duration > cfg.window)].reset_index(drop=True)
print(f"{cfg.detector}: {len(glitches)} low-frequency glitches longer than {cfg.window}s")

# Tile each glitch with ceil(duration / window) windows, centred so any overhang is split evenly.
pieces = []
for gid, g in glitches.iterrows():
    n = int(np.ceil((g["tend"] - g["tstart"]) / cfg.window))
    first = (g["tstart"] + g["tend"]) / 2.0 - n * cfg.window / 2.0
    pieces += [(gid, k, n, first + k * cfg.window) for k in range(n)]
print(f"{len(pieces)} windows to extract")

manifests = common.load_manifests()
transforms = common.build_transforms(common.manifest_rates(manifests) | {cfg.target_sample_rate})
seq_len = int(cfg.window * cfg.target_sample_rate)

strain_out, other_out, witness_out, gps_out = [], [], [], []
meta = {k: [] for k in ("glitch_id", "piece", "n_pieces", "frequency", "snr", "glitch_tstart", "glitch_tend")}
bad_strain = bad_witness = drop_other = 0
cache = {}

for gid, k, n, win_start in tqdm(pieces, desc=f"{cfg.detector}+{cfg.other_detector}"):
    seg_start = win_start - cfg.whiten_psd_length - cfg.pad_seconds
    seg_end = win_start + cfg.window + cfg.pad_seconds

    strain = common.extract_own_strain(manifests["strain"], cache, win_start, transforms)
    if strain is None:
        bad_strain += 1
        continue
    other_strain, witness, reason = common.extract_companions(manifests, cache, seg_start, seg_end, transforms, seq_len)
    if other_strain is None:
        if reason.startswith("witness"):
            bad_witness += 1
        else:
            drop_other += 1
        continue

    g = glitches.loc[gid]
    strain_out.append(strain)
    other_out.append(other_strain)
    witness_out.append(witness)
    gps_out.append(win_start + cfg.window / 2.0)
    for key, val in (("glitch_id", gid), ("piece", k), ("n_pieces", n), ("frequency", g["frequency"]),
                     ("snr", g["snr"]), ("glitch_tstart", g["tstart"]), ("glitch_tend", g["tend"])):
        meta[key].append(val)

common.close_all(cache)

print(f"\nWindows saved: {len(gps_out)} / {len(pieces)} from {len(set(meta['glitch_id']))} glitches")
print(f"Bad strain: {bad_strain}  Bad witness: {bad_witness}  Dropped (other detector): {drop_other}")

common.save_h5(cfg.glitch_long_h5, strain_out, other_out, witness_out, gps_out,
               glitch_id=np.asarray(meta["glitch_id"], dtype=np.int64),
               piece=np.asarray(meta["piece"], dtype=np.int32),
               n_pieces=np.asarray(meta["n_pieces"], dtype=np.int32),
               frequency=np.asarray(meta["frequency"], dtype=np.float64),
               snr=np.asarray(meta["snr"], dtype=np.float32),
               glitch_tstart=np.asarray(meta["glitch_tstart"], dtype=np.float64),
               glitch_tend=np.asarray(meta["glitch_tend"], dtype=np.float64))
