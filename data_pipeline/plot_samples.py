#!/usr/bin/env python3
"""Timeseries and Q-scan plots of loud glitches, signals and blips: one figure of each per sample showing
strain, the other detector's strain and every witness channel.  e.g. python plot_samples.py --n 3 --min_snr 35"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

import config as cfg
from common import decode_names

class_labels = {
    "glitch": "Glitch",
    "glitch_high": "High-frequency glitch",
    "glitch_long": "Long glitch",
    "signal": "Signal",
    "blip": "Blip",
}
event_labels = {"signal": "Coalescence (peak power)"}
qrange = (4, 64)
f_low = 20.0


def load(path):
    with h5py.File(path, "r") as f:
        d = {k: f[k][:] for k in ("strain", "other_strain", "witness", "gps", "snr")}
        d["channel_names"] = decode_names(f["channel_names"][:])
        d["detector"] = decode_names(f["detector"][:1])[0]
        d["other_detector"] = decode_names(f["other_detector"][:1])[0]
    return d


def channels(d, i):
    """(title, data, sample_rate) for every panel of sample i."""
    own_rate = len(d["strain"][i]) / cfg.window  # glitch strain is kept at native rate
    rows = [(f"{d['detector']} strain", d["strain"][i], own_rate),
            (f"{d['other_detector']} strain", d["other_strain"][i], cfg.target_sample_rate)]
    rows += [(cfg.display_name(n), d["witness"][i, c], cfg.target_sample_rate)
             for c, n in enumerate(d["channel_names"])]
    return rows


def event_time(strain, rate, smooth=64):
    power = np.convolve(np.asarray(strain, dtype=np.float64) ** 2, np.ones(smooth) / smooth, mode="same")
    return np.argmax(power) / rate


def plot_timeseries(rows, title, t_event, event_label, path):
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.2 * len(rows)), sharex=True, layout="constrained")
    for ax, (name, data, rate) in zip(axes, rows):
        ax.plot(np.arange(len(data)) / rate, data, color="black" if "strain" in name else "tab:blue", lw=0.9)
        ax.set_title(name, fontsize=13, fontweight="bold", loc="left")
        ax.set_ylabel("Whitened")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axvline(t_event, color="crimson", lw=1.2, ls="--", alpha=0.7)
    axes[0].annotate(event_label, xy=(t_event, axes[0].get_ylim()[1]), xytext=(6, -8), textcoords="offset points",
                     color="crimson", va="top", fontsize=10, fontweight="bold")
    axes[-1].set_xlabel("Time [s]", fontsize=12)
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qscans(rows, title, t_event, path, ncols=3):
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.6 * nrows), squeeze=False,
                             sharex=True, layout="constrained")
    for ax, (name, data, rate) in zip(axes.flat, rows):
        frange = (f_low, rate / 2)
        try:
            # Log-spaced grid keeps each panel ~500x300 instead of millions of cells.
            q = TimeSeries(np.asarray(data, dtype=np.float64), sample_rate=rate).q_transform(
                qrange=qrange, frange=frange, whiten=False, logf=True, fres=300, tres=0.002)
        except Exception as e:
            ax.text(0.5, 0.5, f"Q-transform failed:\n{e}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name, fontsize=11, fontweight="bold", loc="left")
            continue
        mesh = ax.pcolormesh(q.times.value - q.t0.value, q.frequencies.value, q.value.T,
                             cmap="viridis", vmin=0, vmax=25, shading="auto")
        ax.set_yscale("log")
        ax.set_ylim(f_low, q.frequencies.value[-1])
        ax.axvline(t_event, color="white", lw=1.2, ls="--")
        ax.set_title(name, fontsize=11, fontweight="bold", loc="left")
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh, ax=axes, label="Normalized energy", shrink=0.6)
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", default=["glitch", "signal", "blip"], choices=list(class_labels))
    p.add_argument("--n", type=int, default=3, help="samples per class")
    p.add_argument("--min_snr", type=float, default=35.0, help="only pick samples louder than this")
    p.add_argument("--kind", nargs="+", default=["timeseries", "qscan"], choices=["timeseries", "qscan"])
    p.add_argument("--data_dir", default=str(cfg.time_data_dir))
    p.add_argument("--out", default=str(Path(cfg.time_data_dir) / "sample_plots"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    for cls in args.classes:
        path = Path(args.data_dir) / cfg.class_files[cls]
        if not path.exists():
            print(f"Skipping {cls}: {path} not found")
            continue
        d = load(path)
        loud = np.flatnonzero(d["snr"] > args.min_snr)
        if len(loud) == 0:
            print(f"Skipping {cls}: no samples with SNR > {args.min_snr} (max {d['snr'].max():.1f})")
            continue
        picked = rng.choice(loud, size=min(args.n, len(loud)), replace=False)
        print(f"{cls}: plotting {len(picked)} of {len(loud)} samples with SNR > {args.min_snr}")

        out = Path(args.out) / cls
        out.mkdir(parents=True, exist_ok=True)
        for i in picked:
            rows = channels(d, i)
            t_event = event_time(*rows[0][1:])
            title = f"{class_labels[cls]} | GPS {d['gps'][i]:.3f} | SNR {d['snr'][i]:.1f}"
            if "timeseries" in args.kind:
                plot_timeseries(rows, title, t_event, event_labels.get(cls, "Peak power"),
                                out / f"{cls}_{i}_timeseries.png")
            if "qscan" in args.kind:
                plot_qscans(rows, title, t_event, out / f"{cls}_{i}_qscan.png")
    print("Done. Plots saved under", args.out)


if __name__ == "__main__":
    main()
