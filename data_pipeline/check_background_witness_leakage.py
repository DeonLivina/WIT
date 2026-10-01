#!/usr/bin/env python3
"""Check every saved background window in whitened_background_full.h5 for overlap with
witness triggers """

import h5py
import pandas as pd

import common
import config as cfg

background_file = cfg.full_data_dir / cfg.class_files["background"]
out_csv = cfg.full_data_dir / "background_witness_leakage.csv"
channels = cfg.witness_channels

with h5py.File(background_file, "r") as f:
    gps = f["gps"][:]
print(f"Background samples: {len(gps)}")

spans = {}
for channel in channels:
    df = pd.read_csv(common.find_trigger_csv(channel), usecols=["tstart", "tend"])
    spans[channel] = (df["tstart"].to_numpy(), df["tend"].to_numpy())

rows = []
for g in gps:
    lo, hi = g - cfg.window / 2.0, g + cfg.window / 2.0
    hits = [c for c in channels if ((spans[c][0] < hi) & (spans[c][1] > lo)).any()]
    rows.append({"gps": g, "witness_match_count": len(hits), "witness_total": len(channels),
                 "witness_matched_channels": ";".join(hits)})

out = pd.DataFrame(rows).sort_values("gps")
out.to_csv(out_csv, index=False)

n_clean = int((out["witness_match_count"] == 0).sum())
print(f"Wrote {len(out)} rows to {out_csv}")
print(f"Clean: {n_clean}/{len(out)} ({n_clean / len(out):.1%})")
print(out["witness_match_count"].value_counts().sort_index())
