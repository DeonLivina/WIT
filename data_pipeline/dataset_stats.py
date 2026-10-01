"""Sample counts per time bin and SNR distributions for every class in auto_data/<DET>/, plus the
richest non-overlapping time windows for picking an SSL stretch (plots/CSVs in auto_data/plots/)."""

import argparse

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

data_dir = cfg.time_data_root
out_dir = data_dir / "plots"

detectors = ["H1", "L1"]
class_files = {k: cfg.class_files[k] for k in ("background", "glitch", "signal", "blip")}
colors = {"background": "#2a78d6", "glitch": "#eb6834", "signal": "#1baf7a", "blip": "#eda100"}
gps_to_unix = 315964800 - 18  # GPS epoch in unix time, minus leap seconds (valid from 2017)


def gps_to_utc(gps):
    return pd.to_datetime(np.asarray(gps) + gps_to_unix, unit="s")


def load_table():
    rows = []
    for det in detectors:
        for cls, fn in class_files.items():
            path = data_dir / det / fn
            if not path.exists():
                print(f"[missing] {path}")
                continue
            with h5py.File(path, "r") as f:
                gps = f["gps"][:]
                snr = f["snr"][:] if "snr" in f else np.full(len(gps), np.nan)
            rows.append(pd.DataFrame({"detector": det, "class": cls, "gps": gps, "snr": snr}))
    df = pd.concat(rows, ignore_index=True)
    df["utc"] = gps_to_utc(df["gps"])
    return df


def binned_counts(df, hours):
    df = df.assign(bin=df["utc"].dt.floor(f"{hours}h"))
    return df.groupby(["detector", "bin", "class"]).size().unstack(fill_value=0).reindex(columns=list(class_files), fill_value=0)


def top_windows(df, hours, step_min, top, rank_by):
    """Slide a `hours`-long window in `step_min` steps; rank by number of samples (both detectors)."""
    length, step = hours * 3600, step_min * 60
    first = np.floor((df["gps"].min() + gps_to_unix) / step) * step - gps_to_unix  # grid on round UTC times
    starts = np.arange(first, df["gps"].max(), step)
    table = pd.DataFrame({"gps_start": starts, "gps_end": starts + length})
    for det in detectors:
        for cls in class_files:
            g = np.sort(df.loc[(df["detector"] == det) & (df["class"] == cls), "gps"].to_numpy())
            table[f"{det}_{cls}"] = np.searchsorted(g, starts + length) - np.searchsorted(g, starts)
    for cls in class_files:
        table[cls] = sum(table[f"{d}_{cls}"] for d in detectors)
    table["all"] = table[list(class_files)].sum(axis=1)

    # greedy: best window first, then the best one that does not overlap any already chosen
    picked = []
    for i in table.sort_values(rank_by, ascending=False).index:
        s = table.at[i, "gps_start"]
        if all(abs(s - table.at[j, "gps_start"]) >= length for j in picked):
            picked.append(i)
        if len(picked) == top:
            break
    best = table.loc[picked].reset_index(drop=True)
    best.insert(0, "utc_start", gps_to_utc(best["gps_start"]))
    best.insert(1, "utc_end", gps_to_utc(best["gps_end"]))
    return best


def plot_counts(counts, hours, best, path):
    fig, axes = plt.subplots(len(detectors), 1, figsize=(14, 3.5 * len(detectors)), sharex=True, squeeze=False)
    width = pd.Timedelta(hours=hours) * 0.9
    for ax, det in zip(axes[:, 0], detectors):
        c = counts.loc[det] if det in counts.index.get_level_values(0) else counts.iloc[:0]
        bottom = np.zeros(len(c))
        for cls in class_files:
            ax.bar(c.index, c[cls], width=width, bottom=bottom, align="edge", color=colors[cls],
                   label=f"{cls} (n={c[cls].sum():,})")
            bottom += c[cls].to_numpy()
        if len(best):
            ax.axvspan(best.at[0, "utc_start"], best.at[0, "utc_end"], color="#222222", alpha=0.12,
                       label=f"richest {hours} h window")
        ax.set_ylabel(f"{det} samples / {hours} h")
        ax.legend(frameon=False, ncol=len(class_files) + 1, loc="lower left", bbox_to_anchor=(0, 1.0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    axes[-1, 0].set_xlabel("UTC")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_snr(df, path):
    snr = df.dropna(subset=["snr"])
    snr = snr[snr["snr"] > 0]
    classes = [c for c in class_files if c in set(snr["class"])]
    bins = np.logspace(np.log10(snr["snr"].min()), np.log10(snr["snr"].max()), 50)
    fig, axes = plt.subplots(1, len(detectors), figsize=(12, 4), sharey=True, squeeze=False)
    for ax, det in zip(axes[0], detectors):
        for cls in classes:
            v = snr.loc[(snr["detector"] == det) & (snr["class"] == cls), "snr"]
            if len(v):
                # Fraction per bin, so classes of different size are comparable.
                ax.hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), histtype="step", lw=2,
                        color=colors[cls], label=f"{cls} (n={len(v):,}, median {v.median():.1f})")
        ax.set_xscale("log")
        ax.set_title(det, loc="left")
        ax.set_xlabel("SNR")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("fraction of class")
    fig.suptitle("SNR distribution per class (background has no SNR)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=6, help="bin / window length")
    p.add_argument("--step_min", type=int, default=30, help="sliding-window step")
    p.add_argument("--top", type=int, default=10, help="number of non-overlapping windows to report")
    p.add_argument("--rank_by", choices=["all"] + list(class_files), default="all")
    args = p.parse_args()

    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_table()
    counts = binned_counts(df, args.hours)
    counts.to_csv(out_dir / f"counts_per_{args.hours}h.csv")

    best = top_windows(df, args.hours, args.step_min, args.top, args.rank_by)
    best.to_csv(out_dir / f"top_{args.hours}h_windows.csv", index=False)
    print(f"\nTop {len(best)} non-overlapping {args.hours} h windows by '{args.rank_by}':")
    print(best[["utc_start", "gps_start", "gps_end"] + list(class_files) + ["all"]].to_string(index=False))

    plot_counts(counts, args.hours, best, out_dir / f"counts_per_{args.hours}h.png")
    plot_snr(df, out_dir / "snr_distribution.png")


if __name__ == "__main__":
    main()
