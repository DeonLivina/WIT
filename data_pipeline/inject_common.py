"""Shared injection loop for inject_signal.py and inject_gaussians.py: injects a waveform into real
background strain, pairs it with real witness + other-detector data, and removes the used background."""

import copy
import sys
import traceback

import h5py
import numpy as np
import torch
from gwpy.timeseries import TimeSeries
from ml4gw.gw import compute_network_snr, reweight_snrs
from tqdm import tqdm

import common
import config as cfg

sys.path.insert(0, str(cfg.root / "dataset"))
from utils import load_config  # noqa: E402
from injections import _inject_center, _interp_psd  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)
yaml_cfg = load_config(str(cfg.root / "dataset" / "configs" / f"config_{cfg.detector}.yaml"))

rate = cfg.target_sample_rate
f_min = yaml_cfg.general.f_min
f_max = yaml_cfg.general.f_max
psd_length = yaml_cfg.whiten.psd_length
wave_duration = yaml_cfg.general.waveform_duration
pad_seconds = yaml_cfg.whiten.fduration / 2.0

psd_size = int(psd_length * rate)
kernel_size = int(wave_duration * rate)
pad = int(pad_seconds * rate)
window_size = psd_size + kernel_size + 2 * pad
num_freqs = kernel_size // 2 + 1


def run_injection(make_waveform, output_file, n_target, candidate_seed, snr_seed, desc):
    """make_waveform(sig_cfg, raw_inj) -> (1, 1, KERNEL_SIZE) waveform before SNR reweighting."""
    print(f"Executing pipeline on device: {device}")
    manifests = common.load_manifests()
    transforms = common.build_transforms(
        common.manifest_rates(manifests) | {rate},
        fftlength=yaml_cfg.whiten.fftlength, overlap=yaml_cfg.whiten.overlap, average=yaml_cfg.whiten.average,
        fduration=yaml_cfg.whiten.fduration, highpass=f_min, device=device,
    )
    spectral_density, whiten = transforms[rate]

    with h5py.File(cfg.glitch_h5, "r") as f:
        snr_pool = f["snr"][:].astype(np.float64)
    snr_rng = np.random.default_rng(snr_seed)

    with h5py.File(cfg.background_h5, "r") as f:
        bg_gps = f["gps"][:]
    shuffled_gps = bg_gps.copy()
    np.random.default_rng(candidate_seed).shuffle(shuffled_gps)
    print(f"Background windows available: {len(bg_gps)} (target {n_target})")

    strain_out, other_out, witness_out, gps_out, snr_out = [], [], [], [], []
    skipped = dict(bounds=0, whiten=0, witness=0, other=0, wave=0)
    cache = {}
    error_shown = False
    pbar = tqdm(total=n_target, desc=desc)

    for orig_gps in shuffled_gps:
        if len(gps_out) >= n_target:
            break
        win_start = orig_gps - wave_duration / 2.0
        seg_start = win_start - psd_length - pad_seconds
        seg_end = win_start + wave_duration + pad_seconds

        raw_native, s_rate = common.read_channel(manifests["strain"], cache, cfg.detector, None, seg_start, seg_end)
        if raw_native is None:
            skipped["bounds"] += 1
            continue
        if s_rate != rate:
            raw_native = TimeSeries(raw_native, sample_rate=s_rate).resample(rate).value.copy()
        if len(raw_native) < window_size:
            skipped["bounds"] += 1
            continue
        raw = torch.tensor(raw_native[:window_size], dtype=torch.float64, device=device)[None, None, :]
        raw_inj = raw[..., psd_size:]

        try:
            psd = spectral_density(raw[..., :psd_size])
            psd_i = _interp_psd(psd, num_freqs)
            sig_cfg = copy.deepcopy(yaml_cfg)
            sig_cfg.general.sample_rate = rate
            sig_cfg.general.batch_size = 1
            sig_cfg.general.waveform_duration = wave_duration

            waveform = make_waveform(sig_cfg, raw_inj)
            target = torch.as_tensor(snr_rng.choice(snr_pool, size=1), device=device)
            waveform = reweight_snrs(waveform, target, psd_i, rate, highpass=f_min)
            snr = compute_network_snr(waveform, psd_i, rate, highpass=f_min)
            white = whiten(_inject_center(raw_inj.clone(), waveform, kernel_size, pad), psd).view(-1).cpu().numpy()
        except Exception:
            if not error_shown:
                traceback.print_exc()
                error_shown = True
            skipped["wave"] += 1
            continue

        if len(white) != kernel_size or not np.isfinite(white).all():
            skipped["whiten"] += 1
            continue

        other_strain, witness, reason = common.extract_companions(
            manifests, cache, seg_start, seg_end, transforms, kernel_size,
            target_rate=rate, psd_length=psd_length, device=device,
        )
        if other_strain is None:
            skipped["witness" if reason.startswith("witness") else "other"] += 1
            continue

        strain_out.append(white)
        other_out.append(other_strain)
        witness_out.append(witness)
        gps_out.append(orig_gps)
        snr_out.append(float(snr.item()))
        pbar.set_postfix(**skipped)
        pbar.update(1)

    pbar.close()
    common.close_all(cache)
    print(f"\nBuilt {len(gps_out)} / {n_target}. Skipped: {skipped}")
    if not gps_out:
        raise RuntimeError("No examples produced.")

    common.save_h5(output_file, strain_out, other_out, witness_out, gps_out, snr=np.asarray(snr_out))
    common.remove_background_windows(cfg.background_h5, gps_out)
