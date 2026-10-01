"""Shared helpers for the extract/inject scripts: trigger lookup, chunk manifests,
whitening, companion-channel extraction and HDF5 writing."""

import glob
import os

import h5py
import numpy as np
import torch
from gwpy.timeseries import TimeSeries
from ml4gw.transforms import SpectralDensity, Whiten
from tqdm import tqdm

import config as cfg


def find_trigger_csv(channel, detector=cfg.detector, trigger_dir=cfg.trigger_dir):
    matches = sorted(glob.glob(str(trigger_dir / f"{detector}_{channel}_*.csv")))
    if not matches:
        raise FileNotFoundError(f"No trigger CSV for {detector}:{channel} in {trigger_dir}")
    return matches[0]


def decode_names(names):
    return [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]


def clean_non_numerical(data):
    data = np.asarray(data, dtype=np.float64)
    bad = ~np.isfinite(data)
    if bad.all():
        return np.zeros_like(data)
    if bad.any():
        x = np.arange(len(data))
        data[bad] = np.interp(x[bad], x[~bad], data[~bad])
    return data


def build_transforms(rates, fftlength=cfg.whiten_fftlength, overlap=cfg.whiten_overlap,
                     average=cfg.whiten_average, fduration=cfg.whiten_fduration,
                     highpass=cfg.highpass, device="cpu"):
    return {
        rate: (
            SpectralDensity(sample_rate=rate, fftlength=fftlength, overlap=overlap, average=average).to(device),
            Whiten(fduration=fduration, sample_rate=rate, highpass=highpass).to(device),
        )
        for rate in rates
    }


def whiten_block(block, rate, transforms, psd_len, length, target_rate=None, device="cpu"):
    """Whiten block[psd_len:] with the PSD of block[:psd_len], resampling to target_rate if given.
    Returns (array, None) on success or (None, reason)."""
    block = np.asarray(block, dtype=np.float64)
    if len(block) == 0 or np.std(block) == 0:
        return None, "flat"
    try:
        psd_fn, whiten = transforms[rate]
        x = torch.from_numpy(block).to(device).view(1, 1, -1)
        y = whiten(x[..., psd_len:], psd_fn(x[..., :psd_len])).view(-1).cpu().numpy()
        if target_rate and target_rate != rate:
            y = np.asarray(TimeSeries(y, sample_rate=rate).resample(target_rate).value)
    except Exception as e:
        return None, f"exception:{type(e).__name__}"
    if len(y) != length or not np.isfinite(y).all():
        return None, "bad_output"
    return y, None


def load_strain_manifest(detector, strain_dir=cfg.strain_dir):
    manifest = {}
    for path in sorted(glob.glob(str(strain_dir / f"{detector[0]}-{detector}_*.hdf5"))):
        with h5py.File(path, "r") as f:
            start = int(f["meta"]["GPSstart"][()])
            duration = int(f["meta"]["Duration"][()])
            rate = round(1.0 / f["strain"]["Strain"].attrs["Xspacing"])
        manifest.setdefault(start, (start, start + duration, rate, path))
    return sorted(manifest.values())


def load_witness_manifest(detector, channel, witness_dir=cfg.witness_dir):
    manifest = []
    for path in sorted(glob.glob(str(witness_dir / f"{detector}_{channel}_*.hdf5"))):
        with h5py.File(path, "r") as f:
            dset = f[f"{detector}:{channel}"]
            rate = round(1.0 / dset.attrs["dx"])
            x0 = float(dset.attrs["x0"])
            manifest.append((x0, x0 + dset.shape[0] / rate, rate, path))
    if not manifest:
        raise RuntimeError(f"No witness chunks found for {detector}:{channel}")
    return sorted(manifest)


def load_manifests():
    m = {
        "strain": load_strain_manifest(cfg.detector),
        "other_strain": load_strain_manifest(cfg.other_detector),
        "witness": {c: load_witness_manifest(cfg.detector, c) for c in cfg.witness_channels},
        "other_witness": {c: load_witness_manifest(cfg.other_detector, c) for c in cfg.other_witness_channels},
    }
    print(f"Strain chunks: {cfg.detector}={len(m['strain'])}, {cfg.other_detector}={len(m['other_strain'])}")
    return m


def manifest_rates(manifests):
    chunks = manifests["strain"] + manifests["other_strain"]
    for group in ("witness", "other_witness"):
        for man in manifests[group].values():
            chunks += man
    return {rate for _, _, rate, _ in chunks}


def find_chunk(manifest, start, end):
    for chunk_start, chunk_end, rate, path in manifest:
        if chunk_start <= start and end <= chunk_end:
            return chunk_start, rate, path
    return None


def read_channel(manifest, cache, detector, channel, start, end):
    """Raw samples over [start, end) from a single chunk; channel=None means strain."""
    chunk = find_chunk(manifest, start, end)
    if chunk is None:
        return None, None
    chunk_start, rate, path = chunk
    if path not in cache:
        cache[path] = h5py.File(path, "r")
    dset = cache[path]["strain"]["Strain"] if channel is None else cache[path][f"{detector}:{channel}"]
    s0 = int(round((start - chunk_start) * rate))
    s1 = int(round((end - chunk_start) * rate))
    return clean_non_numerical(dset[s0:s1]), rate


def extract_channel(manifest, cache, detector, channel, start, end, transforms, length,
                    target_rate=cfg.target_sample_rate, psd_length=cfg.whiten_psd_length, device="cpu"):
    raw, rate = read_channel(manifest, cache, detector, channel, start, end)
    if raw is None:
        return None, "bad_bounds"
    return whiten_block(raw, rate, transforms, int(psd_length * rate), length, target_rate, device)


def get_strain_segment(manifest, cache, start, end, rate):
    """Strain over [start, end), stitched across chunk boundaries with zero-filled gaps."""
    out = np.zeros(int(round((end - start) * rate)), dtype=np.float64)
    for chunk_start, chunk_end, chunk_rate, path in manifest:
        lo, hi = max(start, chunk_start), min(end, chunk_end)
        if lo >= hi:
            continue
        if path not in cache:
            cache[path] = h5py.File(path, "r")
        s0 = int(round((lo - chunk_start) * chunk_rate))
        s1 = int(round((hi - chunk_start) * chunk_rate))
        o0 = int(round((lo - start) * chunk_rate))
        data = clean_non_numerical(cache[path]["strain"]["Strain"][s0:s1])
        out[o0:o0 + len(data)] = data[:len(out) - o0]
    return out


def extract_own_strain(manifest, cache, win_start, transforms):
    """Whitened own strain for [win_start, win_start + window) at its native rate, or None."""
    seg_start = win_start - cfg.whiten_psd_length - cfg.pad_seconds
    seg_end = win_start + cfg.window + cfg.pad_seconds
    rate = next((r for s, e, r, _ in manifest if s <= win_start < e), cfg.target_sample_rate)
    raw = get_strain_segment(manifest, cache, seg_start, seg_end, rate)
    strain, _ = whiten_block(raw, rate, transforms, int(cfg.whiten_psd_length * rate), int(cfg.window * rate))
    return strain


def extract_companions(manifests, cache, start, end, transforms, length, **kw):
    """Own witness + other detector's strain and witness (real data) over [start, end).
    Returns (other_strain, witness[C, T], None) or (None, None, reason)."""
    witness = []
    for c in cfg.witness_channels:
        w, reason = extract_channel(manifests["witness"][c], cache, cfg.detector, c, start, end, transforms, length, **kw)
        if w is None:
            return None, None, f"witness:{c}:{reason}"
        witness.append(w)
    other_strain, reason = extract_channel(manifests["other_strain"], cache, cfg.other_detector, None,
                                           start, end, transforms, length, **kw)
    if other_strain is None:
        return None, None, f"other_strain:{reason}"
    for c in cfg.other_witness_channels:
        w, reason = extract_channel(manifests["other_witness"][c], cache, cfg.other_detector, c,
                                    start, end, transforms, length, **kw)
        if w is None:
            return None, None, f"other_witness:{c}:{reason}"
        witness.append(w)
    return other_strain, np.stack(witness), None


def close_all(cache):
    for f in cache.values():
        f.close()


def save_h5(path, strain, other_strain, witness, gps, **extra):
    if len(gps) == 0:
        print("No samples for", path, "- skipping")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, arr in (("strain", strain), ("other_strain", other_strain), ("witness", witness)):
            f.create_dataset(name, data=np.asarray(arr, dtype=np.float32), compression="gzip")
        f.create_dataset("gps", data=np.asarray(gps, dtype=np.float64))
        for name, arr in extra.items():
            f.create_dataset(name, data=np.asarray(arr))
        f.create_dataset("channel_names", data=np.array(cfg.channel_names, dtype="S"))
        f.create_dataset("detector", data=np.array([cfg.detector] * len(gps), dtype="S"))
        f.create_dataset("other_detector", data=np.array([cfg.other_detector] * len(gps), dtype="S"))
    print(f"Saved: {path}  strain {np.shape(strain)}  witness {np.shape(witness)}")


def extract_glitch_bands(glitches, low_path, high_path, max_low=None, jitter_seed=44, jitter_margin=0.02):
    """Whiten each glitch (rows with tstart, tend, frequency, snr) with its real witness and other-detector
    data; peak frequency <= Nyquist goes to low_path (at most max_low of them), the rest to high_path."""
    nyquist = cfg.target_sample_rate / 2.0
    manifests = load_manifests()
    transforms = build_transforms(manifest_rates(manifests) | {cfg.target_sample_rate})
    seq_len = int(cfg.window * cfg.target_sample_rate)

    gtstart = glitches["tstart"].to_numpy()
    gtend = glitches["tend"].to_numpy()
    gtime = (gtstart + gtend) / 2.0
    gfreq = glitches["frequency"].to_numpy()
    gsnr = glitches["snr"].to_numpy()

    # Short glitches get a random window position that still fully contains them; others are centred.
    gdur = gtend - gtstart
    can_jitter = (gdur < 0.6) & (cfg.window - gdur - 2 * jitter_margin > 0)
    win_starts = gtime - cfg.window / 2.0
    rng = np.random.default_rng(jitter_seed)
    win_starts[can_jitter] = rng.uniform(gtend[can_jitter] + jitter_margin - cfg.window,
                                         gtstart[can_jitter] - jitter_margin)

    out = {k: {"strain": [], "other": [], "witness": [], "gps": [], "freq": [], "snr": []} for k in ("low", "high")}
    bad_strain = bad_witness = drop_other = 0
    cache = {}

    for t, fr, snr, win_start in tqdm(zip(gtime, gfreq, gsnr, win_starts), total=len(gtime),
                                      desc=f"{cfg.detector}+{cfg.other_detector}"):
        band = "low" if fr <= nyquist else "high"
        if band == "low" and max_low is not None and len(out["low"]["gps"]) >= max_low:
            continue

        seg_start = win_start - cfg.whiten_psd_length - cfg.pad_seconds
        seg_end = win_start + cfg.window + cfg.pad_seconds

        strain = extract_own_strain(manifests["strain"], cache, win_start, transforms)
        if strain is None:
            bad_strain += 1
            continue

        other_strain, witness, reason = extract_companions(manifests, cache, seg_start, seg_end, transforms, seq_len)
        if other_strain is None:
            if reason.startswith("witness"):
                bad_witness += 1
            else:
                drop_other += 1
            continue

        o = out[band]
        o["strain"].append(strain)
        o["other"].append(other_strain)
        o["witness"].append(witness)
        o["gps"].append(t)
        o["freq"].append(fr)
        o["snr"].append(snr)

    close_all(cache)

    print(f"\nLow-freq: {len(out['low']['gps'])}  High-freq: {len(out['high']['gps'])}")
    print(f"Bad strain: {bad_strain}  Bad witness: {bad_witness}  Dropped (other detector): {drop_other}")

    for band, path in (("low", low_path), ("high", high_path)):
        o = out[band]
        save_h5(path, o["strain"], o["other"], o["witness"], o["gps"],
                frequency=np.asarray(o["freq"], dtype=np.float64),
                snr=np.asarray(o["snr"], dtype=np.float32))


def remove_background_windows(path, used_gps):
    with h5py.File(path, "r") as f:
        data = {k: f[k][:] for k in f}
    keep = ~np.isin(data["gps"], np.asarray(used_gps, dtype=np.float64))
    tmp = str(path) + ".tmp"
    with h5py.File(tmp, "w") as f:
        for k, v in data.items():
            if k == "channel_names":
                f.create_dataset(k, data=v)
            elif v.ndim > 1:
                f.create_dataset(k, data=v[keep], compression="gzip")
            else:
                f.create_dataset(k, data=v[keep])
    os.replace(tmp, path)
    print(f"Removed {int((~keep).sum())} used windows from {path}; {int(keep.sum())} remain")
