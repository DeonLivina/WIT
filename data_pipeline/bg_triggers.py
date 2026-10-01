#!/usr/bin/env python3
"""Tile 1 s background windows into gaps between strain triggers, then drop any window overlapping
a strain or witness trigger """

import os

import numpy as np
import pandas as pd

import common
import config as cfg

buffer_seconds = 1.0
max_glitch_duration = 1.0


def overlaps_any(win_start, win_end, tstart, tend):
    """True where [win_start, win_end) overlaps any [tstart, tend) interval (sorted + running max of tend)."""
    order = np.argsort(tstart)
    tstart_sorted = tstart[order]
    max_tend = np.maximum.accumulate(tend[order])
    hi = np.searchsorted(tstart_sorted, win_end, side="left")
    idx = np.clip(hi - 1, 0, len(tstart_sorted) - 1)
    return (hi > 0) & (max_tend[idx] > win_start)


triggers = pd.read_csv(common.find_trigger_csv("GDS-CALIB_STRAIN")).sort_values("tstart")
tstart = triggers["tstart"].to_numpy()
tend = triggers["tend"].to_numpy()
durations = tend - tstart
print(f"{cfg.detector}: {len(tstart)} strain triggers")

windows = []
for i in range(len(tstart) - 1):
    if durations[i] > max_glitch_duration or durations[i + 1] > max_glitch_duration:
        continue
    gap_start = tend[i] + buffer_seconds
    gap_end = tstart[i + 1] - buffer_seconds
    for j in range(int(np.floor((gap_end - gap_start) / cfg.window))):
        s = gap_start + j * cfg.window
        windows.append([s, s + cfg.window, s + cfg.window / 2.0])

bg = pd.DataFrame(windows, columns=["gps_start", "gps_end", "gps_center"])
print(f"candidate windows after gap tiling: {len(bg)}")

ws, we = bg["gps_start"].to_numpy(), bg["gps_end"].to_numpy()
keep = ~overlaps_any(ws, we, tstart, tend)
print(f"  dropped (strain overlap): {int((~keep).sum())}")
for channel in cfg.witness_channels:
    w = pd.read_csv(common.find_trigger_csv(channel), usecols=["tstart", "tend"])
    hit = overlaps_any(ws, we, w["tstart"].to_numpy(), w["tend"].to_numpy())
    print(f"  dropped ({channel} overlap): {int(hit.sum())}")
    keep &= ~hit

bg = bg[keep].reset_index(drop=True)
print(f"\n{cfg.detector}: clean background windows: {len(bg)}")

os.makedirs(cfg.full_data_dir, exist_ok=True)
bg.to_csv(cfg.background_triggers_csv, index=False)
print("Saved:", cfg.background_triggers_csv)
