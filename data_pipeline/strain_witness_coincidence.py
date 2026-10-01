#!/usr/bin/env python3
"""Keep strain Omicron triggers that have at least one witness trigger within that channel's HVeto
`twin` window -> auto_data/<DET>/strain_witness_coincidence.csv."""

import os

import pandas as pd

import common
import config as cfg

strain_snr_threshold = 7
witness_snr_threshold = 7
default_tolerance = 0.10

# HVeto `twin` windows (s) per witness channel.
witness_tolerances = {
    "ISI-HAM4_BLND_GS13Z_IN1_DQ": 0.80,
    "LSC-POP_A_LF_OUT_DQ": 0.10,
    "LSC-REFL_A_LF_OUT_DQ": 0.20,
    "LSC-REFL_A_RF45_I_ERR_DQ": 0.80,
    "LSC-REFL_A_RF9_Q_ERR_DQ": 0.10,
    "PEM-EY_VMON_ETMY_ESDPOWER24_DQ": 0.10,
    "ISI-HAM6_BLND_GS13RZ_IN1_DQ": 1.00,
    "PEM-EY_ACC_BEAMTUBE_MAN_Y_DQ": 1.00,
    "ASC-CHARD_P_OUT_DQ": 0.20,
}

strain = pd.read_csv(common.find_trigger_csv("GDS-CALIB_STRAIN"))
strain = strain[strain["snr"] >= strain_snr_threshold].reset_index(drop=True)
strain["_row_id"] = strain.index
print(f"{cfg.detector}: {len(strain)} strain triggers with SNR >= {strain_snr_threshold}")

left = strain[["_row_id", "time"]].sort_values("time")
matched_cols = []
for channel in cfg.witness_channels:
    w = pd.read_csv(common.find_trigger_csv(channel), usecols=["time", "snr"])
    w = w[w["snr"] >= witness_snr_threshold].sort_values("time")
    w["_hit"] = w["time"]
    merged = pd.merge_asof(left, w, on="time", direction="nearest",
                           tolerance=witness_tolerances.get(channel, default_tolerance)).set_index("_row_id")
    col = f"witness_{channel}_matched"
    strain[col] = strain["_row_id"].map(merged["_hit"].notna())
    matched_cols.append(col)

strain["witness_match_count"] = strain[matched_cols].sum(axis=1)
strain["witness_matched_channels"] = strain[matched_cols].apply(
    lambda row: ";".join(ch for ch, col in zip(cfg.witness_channels, matched_cols) if row[col]), axis=1)

base_cols = ["time", "frequency", "tstart", "tend", "fstart", "fend", "snr", "q", "amplitude", "phase"]
out = strain[strain["witness_match_count"] > 0].sort_values("time")[
    base_cols + ["witness_match_count", "witness_matched_channels"]]

os.makedirs(cfg.full_data_dir, exist_ok=True)
out.to_csv(cfg.coincidence_csv, index=False)
print(f"Wrote {len(out)} / {len(strain)} coincident triggers to {cfg.coincidence_csv}")
print(out["witness_match_count"].value_counts().sort_index())
