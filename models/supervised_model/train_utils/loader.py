"""Time-aligned 4-class loader (background / glitch / signal / blip) from auto_data/<DET>/.

x (N, T, 11), channels last. Only the primary detector's strain, never both:
    0      primary detector strain (H1 strain for H1 samples, L1 strain for L1 samples)
    1-5    H1 witnesses
    6-10   L1 witnesses    (fixed order for every sample)
one_detector_only=True gives x (N, T, 6): strain + the primary detector's own 5 witnesses.
Everything from TEST_TIME_CUTOFF_GPS (2019-05-05 00:00 UTC, the last day) onwards goes to test.
Batches: (x, y_class, y_detector, is_special), plus snr (NaN for background) with with_snr=True.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[3]  # repo root (train_utils -> supervised_model -> models -> root)
DATA_DIR = Path(os.environ.get("GW_OUT_ROOT", ROOT / "auto_data"))  # same folder as time_aligned_pipeline

LABEL_NAMES = {0: "background", 1: "glitch", 2: "signal", 3: "blip"}
DETECTOR_NAMES = {0: "H1", 1: "L1"}
DETECTOR_LABELS = {"H1": 0, "L1": 1}
FILES = {
    0: "whitened_background_full.h5",
    1: "whitened_glitches.h5",
    2: "whitened_signals_full.h5",
    3: "whitened_gaussians.h5",
}
TEST_GLITCH_FILE = "all_test_glitches.h5"  # every strain-trigger glitch in the test period (extract_test_glitches.py)

H1_WITNESS_CHANNELS = [
    "ISI-HAM4_BLND_GS13Z_IN1_DQ",
    "LSC-POP_A_LF_OUT_DQ",
    "LSC-REFL_A_LF_OUT_DQ",
    "LSC-REFL_A_RF45_I_ERR_DQ",
    "LSC-REFL_A_RF9_Q_ERR_DQ",
]
L1_WITNESS_CHANNELS = [
    "LSC-POP_A_LF_OUT_DQ",
    "PEM-EY_VMON_ETMY_ESDPOWER24_DQ",
    "ISI-HAM6_BLND_GS13RZ_IN1_DQ",
    "PEM-EY_ACC_BEAMTUBE_MAN_Y_DQ",
    "ASC-CHARD_P_OUT_DQ",
]
WITNESS_CHANNELS = [f"H1:{c}" for c in H1_WITNESS_CHANNELS] + [f"L1:{c}" for c in L1_WITNESS_CHANNELS]

N_BG_PER_DETECTOR = 1500
CLASS_TOTALS = {1: 3000, 2: 3000, 3: 3000}   # glitch / signal / blip: all of L1, backfilled from H1
TRAIN_FRAC, VAL_FRAC = 2200 / 3000, 300 / 3000
WINDOW = 1.0


def utc_to_gps(*utc, leap_offset=18):
    epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    return (datetime(*utc, tzinfo=timezone.utc) - epoch).total_seconds() + leap_offset


TEST_TIME_CUTOFF_GPS = utc_to_gps(2019, 5, 5)             # 2019-05-05 00:00 UTC
SPECIAL_EVENT_GPS = utc_to_gps(2019, 5, 5, 15, 10, 38)  # loudest non-signal (glitch) event on record


def own_witness_channels(detector):
    return [f"{detector}:{c}" for c in {"H1": H1_WITNESS_CHANNELS, "L1": L1_WITNESS_CHANNELS}[detector]]


def uses_one_detector(module):
    """True if a trained module expects only the primary detector's 5 witnesses."""
    return module.hparams.n_witness == len(H1_WITNESS_CHANNELS)


def load_file(label, detector, one_detector_only=False):
    return load_path(DATA_DIR / detector / FILES[label], detector, one_detector_only)


def load_path(path, detector, one_detector_only=False, extra=()):
    """(x, gps, snr, *extra datasets) for one file, bad samples removed; None if the file is missing."""
    if not path.exists():
        print(f"[missing] {path}")
        return None
    with h5py.File(path, "r") as f:
        strain, witness, gps = f["strain"][:], f["witness"][:], f["gps"][:]
        snr = f["snr"][:] if "snr" in f else np.full(len(gps), np.nan)
        names = [n.decode() for n in f["channel_names"][:]]
        extras = [f[k][:] for k in extra]

    T = witness.shape[-1]
    if strain.shape[-1] != T:  # glitch strain is saved at the native rate
        strain = resample_poly(strain, T, strain.shape[-1], axis=-1)
    channels = own_witness_channels(detector) if one_detector_only else WITNESS_CHANNELS
    witness = witness[:, [names.index(c) for c in channels]]

    x = np.concatenate([strain[:, None], witness], axis=1).transpose(0, 2, 1).astype(np.float32)
    ok = np.isfinite(x).all((1, 2)) & (np.abs(x) < 1e4).all((1, 2)) & (strain != 0).any(1)
    return (x[ok], gps[ok], snr[ok], *(e[ok] for e in extras))


def drop_leaky_background(x, gps, snr, detector):
    path = DATA_DIR / detector / "background_witness_leakage.csv"
    if not path.exists():
        return x, gps, snr
    df = pd.read_csv(path)
    bad = np.isin(gps, df.loc[df["witness_match_count"] > 0, "gps"])
    return x[~bad], gps[~bad], snr[~bad]


def gather(label, rng, one_detector_only=False):
    data = {d: load_file(label, d, one_detector_only) for d in DETECTOR_LABELS}
    if label == 0:
        data = {d: drop_leaky_background(*v, d) for d, v in data.items()}
        take = {"H1": N_BG_PER_DETECTOR, "L1": N_BG_PER_DETECTOR}
    else:
        n_l1 = 0 if data["L1"] is None else min(len(data["L1"][0]), CLASS_TOTALS[label])
        take = {"L1": n_l1, "H1": CLASS_TOTALS[label] - n_l1}

    xs, ds, gs, ss = [], [], [], []
    for d, n in take.items():
        if data[d] is None or n == 0:
            continue
        x, gps, snr = data[d]
        sel = rng.permutation(len(x))[:n]
        xs.append(x[sel])
        gs.append(gps[sel])
        ss.append(snr[sel])
        ds.append(np.full(len(sel), DETECTOR_LABELS[d], dtype=np.int64))
    print(f"{LABEL_NAMES[label]}: " + " + ".join(f"{len(g)} {DETECTOR_NAMES[d[0]]}" for g, d in zip(gs, ds)))
    x = np.concatenate(xs)
    return x, np.full(len(x), label, dtype=np.int64), np.concatenate(ds), np.concatenate(gs), np.concatenate(ss)


def time_groups(gps):
    """Samples within WINDOW of each other share a group (the H1 and L1 sample of the
    same time carry identical witness channels, so a group stays in one split)."""
    order = np.argsort(gps)
    gid = np.empty(len(gps), dtype=np.int64)
    gid[order] = np.cumsum(np.r_[0, np.diff(gps[order]) > WINDOW])
    return gid


def split_class(gps, rng, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, test_cutoff_gps=TEST_TIME_CUTOFF_GPS):
    """Groups at/after the cutoff (or holding the special event) go to test, topped up
    with random earlier groups; the rest is split train/val."""
    n = len(gps)
    n_test = n - round(n * train_frac) - round(n * val_frac)
    special = np.abs(gps - SPECIAL_EVENT_GPS) <= WINDOW / 2
    gid = time_groups(gps)
    sizes = np.bincount(gid)

    forced = np.unique(gid[(gps >= test_cutoff_gps) | special])
    free = rng.permutation(np.setdiff1d(np.arange(len(sizes)), forced))
    need = n_test - sizes[forced].sum()
    print(f"  {sizes[forced].sum()} samples from {len(gps)} forced to test (target {n_test})")
    n_fill = np.searchsorted(np.cumsum(sizes[free]), need) + 1 if need > 0 else 0

    test_g, rest = np.concatenate([forced, free[:n_fill]]), free[n_fill:]
    n_train_g = round(len(rest) * train_frac / (train_frac + val_frac))
    splits = [rest[:n_train_g], rest[n_train_g:], test_g]
    return [np.flatnonzero(np.isin(gid, g)) for g in splits], special


def get_dataloaders(batch_size=32, seed=42, with_snr=False, one_detector_only=False, gather_fn=gather,
                    train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, test_cutoff_gps=TEST_TIME_CUTOFF_GPS):
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for label in LABEL_NAMES:
        x, y, d, gps, snr = gather_fn(label, rng, one_detector_only)
        idx, special = split_class(gps, rng, train_frac, val_frac, test_cutoff_gps)
        for name, i in zip(parts, idx):
            parts[name].append((x[i], y[i], d[i], special[i], snr[i]))

    datasets = {}
    for name, p in parts.items():
        x, y, d, s, snr = (np.concatenate(a) for a in zip(*p))
        arrays = (x, y, d, s) + ((snr.astype(np.float32),) if with_snr else ())
        datasets[name] = TensorDataset(*(torch.from_numpy(a) for a in arrays))
        print(f"{name}: {len(y)} samples, {int(s.sum())} special")

    y_train = datasets["train"].tensors[1].numpy()
    weights = 1.0 / np.bincount(y_train, minlength=len(LABEL_NAMES)).clip(min=1)
    sampler = WeightedRandomSampler(torch.from_numpy(weights[y_train]).float(), len(y_train), replacement=True)

    train = DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler, pin_memory=True)
    val = DataLoader(datasets["val"], batch_size=batch_size, pin_memory=True)
    test = DataLoader(datasets["test"], batch_size=batch_size, pin_memory=True)
    witness_names = WITNESS_CHANNELS
    if one_detector_only:
        witness_names = [f"own witness {i + 1}" for i in range(len(H1_WITNESS_CHANNELS))]
    meta = {
        "num_classes": len(LABEL_NAMES),
        "n_witness": len(witness_names),
        "channel_names": ["strain"] + witness_names,
        "one_detector_only": one_detector_only,
        "label_names": LABEL_NAMES,
        "detector_names": DETECTOR_NAMES,
    }
    return train, val, test, meta


def get_test_glitch_loader(batch_size=32, one_detector_only=False, test_cutoff_gps=TEST_TIME_CUTOFF_GPS):
    """All glitches of TEST_GLITCH_FILE from both detectors (only those at/after test_cutoff_gps, so none can
    have been trained on), labelled glitch, batches as in get_dataloaders(with_snr=True). None if no file exists."""
    glitch = {v: k for k, v in LABEL_NAMES.items()}["glitch"]
    xs, ds, gs, ss, ms = [], [], [], [], []
    for det, det_id in DETECTOR_LABELS.items():
        data = load_path(DATA_DIR / det / TEST_GLITCH_FILE, det, one_detector_only)
        if data is None:
            continue
        x, gps, snr = data
        keep = gps >= test_cutoff_gps
        print(f"test glitches {det}: {int(keep.sum())}" + (f" ({int((~keep).sum())} before the test cutoff dropped)"
                                                           if not keep.all() else ""))
        xs.append(x[keep])
        gs.append(gps[keep])
        ss.append(snr[keep])
        ds.append(np.full(int(keep.sum()), det_id, dtype=np.int64))
        ms.append(witness_matched(gps[keep], det))
    if not xs or sum(len(g) for g in gs) == 0:
        return None

    x, d, gps, snr = (np.concatenate(a) for a in (xs, ds, gs, ss))
    special = np.abs(gps - SPECIAL_EVENT_GPS) <= WINDOW / 2
    dataset = TensorDataset(*(torch.from_numpy(a) for a in (
        x, np.full(len(x), glitch, dtype=np.int64), d, special, snr.astype(np.float32))))
    dl = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    # In batch order (no shuffling): True/False if the glitch has an own-detector witness coincidence, None if unknown.
    dl.witness_matched = None if any(m is None for m in ms) else np.concatenate(ms)
    return dl


def witness_matched(gps, detector):
    """Which glitch times (trigger midpoints) are in the pipeline's strain_witness_coincidence.csv; None if missing."""
    path = DATA_DIR / detector / "strain_witness_coincidence.csv"
    if not path.exists():
        print(f"[missing] {path}: no witness-matched breakdown")
        return None
    c = pd.read_csv(path, usecols=["tstart", "tend"])
    return np.isin(np.round(gps, 4), np.round((c["tstart"] + c["tend"]).to_numpy() / 2.0, 4))
