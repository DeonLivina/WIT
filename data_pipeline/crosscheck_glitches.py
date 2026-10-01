#!/usr/bin/env python3
"""Match each GravitySpy catalog glitch (CATALOG_CSV, must be supplied) to strain and witness Omicron
triggers -> auto_data/<DET>/glitch_witness_crosscheck.csv."""

import os

import pandas as pd

import common
import config as cfg

catalog_csv = cfg.root / f"{cfg.detector}_O3.csv"
out_csv = cfg.full_data_dir / "glitch_witness_crosscheck.csv"
channels = cfg.witness_channels

conf_threshold = 0.60
max_duration = 1.0
strain_tolerance = 0.0
witness_tolerance = 0.5
ignore_classes = {"No_Glitch", "None_of_the_Above", "Blip", "Blip_Low_Frequency", "Chirp"}


def match_nearest(events, channel, tolerance):
    """Nearest trigger within `tolerance` of each event's peak_time, index-aligned to events."""
    trig = pd.read_csv(common.find_trigger_csv(channel), usecols=["time", "snr", "tstart", "tend"]).sort_values("time")
    left = events[["peak_time"]].reset_index().rename(columns={"index": "_row_id"}).sort_values("peak_time")
    return pd.merge_asof(left, trig, left_on="peak_time", right_on="time",
                         direction="nearest", tolerance=tolerance).set_index("_row_id")


df = pd.read_csv(catalog_csv)
df = df[(df["ml_confidence"] >= conf_threshold) & (df["duration"] <= max_duration)
        & ~df["ml_label"].isin(ignore_classes)].copy()
df["peak_time"] = df["peak_time"].astype(float) + df["peak_time_ns"].astype(float) * 1e-9
print(f"{len(df)} candidate glitches after filtering")

m = match_nearest(df, "GDS-CALIB_STRAIN", strain_tolerance)
df["strain_matched"] = m["time"].notna()
for field in ("snr", "tstart", "tend"):
    df[f"strain_trigger_{field}"] = m[field]

witness_cols = []
for channel in channels:
    m = match_nearest(df, channel, witness_tolerance)
    df[f"witness_{channel}_matched"] = m["time"].notna()
    for field in ("snr", "tstart", "tend"):
        df[f"witness_{channel}_{field}"] = m[field]
    witness_cols += [f"witness_{channel}_{f}" for f in ("matched", "snr", "tstart", "tend")]

matched = [f"witness_{c}_matched" for c in channels]
df["witness_match_count"] = df[matched].sum(axis=1)
df["witness_total"] = len(channels)
df["witness_matched_channels"] = df[matched].apply(
    lambda row: ";".join(c for c, col in zip(channels, matched) if row[col]), axis=1)

keep_cols = ["peak_time", "ml_label", "ml_confidence", "duration", "snr",
             "strain_matched", "strain_trigger_snr", "strain_trigger_tstart", "strain_trigger_tend",
             "witness_match_count", "witness_total", "witness_matched_channels"] + witness_cols
out = df.sort_values("peak_time")[keep_cols]

os.makedirs(cfg.full_data_dir, exist_ok=True)
out.to_csv(out_csv, index=False)
print(f"Wrote {len(out)} rows to {out_csv}")
print(f"strain match rate: {out['strain_matched'].mean():.1%}")
print(out["witness_match_count"].value_counts().sort_index())
