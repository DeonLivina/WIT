"""Shared settings for the time-aligned pipeline: detector, paths, witness channels, whitening.
Output goes to auto_data/<detector>/; GW_DETECTOR and GW_OUT_ROOT env vars override detector and folder."""

import os
from datetime import datetime, timezone
from pathlib import Path

detector = os.environ.get("GW_DETECTOR", "H1")
other_detector = {"H1": "L1", "L1": "H1"}[detector]

root = Path(__file__).resolve().parent.parent
strain_dir = root / "data" / "og"
witness_dir = root / "data" / "witness_data"
trigger_dir = root / f"triggers_{detector}"

# Every generated file goes to auto_data/<detector>/ (or <GW_OUT_ROOT>/<detector>/).
time_data_root = Path(os.environ.get("GW_OUT_ROOT", root / "auto_data"))
full_data_dir = time_data_dir = time_data_root / detector

background_triggers_csv = full_data_dir / "background_triggers.csv"
coincidence_csv = full_data_dir / "strain_witness_coincidence.csv"

class_files = {
    "background": "whitened_background_full.h5",
    "glitch": "whitened_glitches.h5",
    "glitch_high": "whitened_high_glitches.h5",
    "glitch_long": "whitened_long_glitches.h5",
    "signal": "whitened_signals_full.h5",
    "blip": "whitened_gaussians.h5",
    "test_glitch": "all_test_glitches.h5",
    "test_glitch_high": "all_test_high_glitches.h5",
}
background_h5 = time_data_dir / class_files["background"]
glitch_h5 = time_data_dir / class_files["glitch"]
glitch_high_h5 = time_data_dir / class_files["glitch_high"]
glitch_long_h5 = time_data_dir / class_files["glitch_long"]
signal_h5 = time_data_dir / class_files["signal"]
blip_h5 = time_data_dir / class_files["blip"]
test_glitch_h5 = time_data_dir / class_files["test_glitch"]
test_glitch_high_h5 = time_data_dir / class_files["test_glitch_high"]


def utc_to_gps(utc, leap_offset=18):
    t = datetime.fromisoformat(utc).replace(tzinfo=timezone.utc)
    return (t - datetime(1980, 1, 6, tzinfo=timezone.utc)).total_seconds() + leap_offset


# Start of the held-out test period (keep equal to data.test_cutoff_utc in models/supervised_model/config.yaml).
test_cutoff_utc = os.environ.get("GW_TEST_CUTOFF_UTC", "2019-05-05 00:00:00")
test_cutoff_gps = utc_to_gps(test_cutoff_utc)

# Witness channel -> readable name used in plots.
witness_by_detector = {
    "H1": {
        "ISI-HAM4_BLND_GS13Z_IN1_DQ": "HAM4 Seismic Isolation (vertical)",
        "LSC-POP_A_LF_OUT_DQ": "Power Recycling Cavity Power",
        "LSC-REFL_A_LF_OUT_DQ": "Reflected Power",
        "LSC-REFL_A_RF45_I_ERR_DQ": "Reflected RF45 Error (I)",
        "LSC-REFL_A_RF9_Q_ERR_DQ": "Reflected RF9 Error (Q)",
    },
    "L1": {
        "LSC-POP_A_LF_OUT_DQ": "Power Recycling Cavity Power",
        "PEM-EY_VMON_ETMY_ESDPOWER24_DQ": "End-Y Test Mass ESD Power Monitor",
        "ISI-HAM6_BLND_GS13RZ_IN1_DQ": "HAM6 Seismic Isolation (yaw)",
        "PEM-EY_ACC_BEAMTUBE_MAN_Y_DQ": "End-Y Beam Tube Accelerometer",
        "ASC-CHARD_P_OUT_DQ": "Common Hard Pitch (arm cavity alignment)",
    },
}
witness_channels = list(witness_by_detector[detector])
other_witness_channels = list(witness_by_detector[other_detector])
channel_names = ([f"{detector}:{c}" for c in witness_channels]
                 + [f"{other_detector}:{c}" for c in other_witness_channels])


def display_name(full_name):
    """Readable name for a "<DET>:<CHANNEL>" witness channel."""
    det, channel = full_name.split(":", 1)
    return f"{det} {witness_by_detector[det][channel]}"


window = 1.0
target_sample_rate = 4096

whiten_fftlength = 2.0
whiten_overlap = None
whiten_average = "median"
whiten_psd_length = 16.0
whiten_fduration = 2.0
highpass = 20.0
pad_seconds = whiten_fduration / 2.0
