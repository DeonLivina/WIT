"""Overlay glitch efficiency vs deadtime curves of several models on the same split.

Efficiency = fraction of glitches flagged as glitch; deadtime = fraction of the split's signal injections flagged
as glitch. The glitches are the whole test-period population (all_test_glitches.h5 of both detectors, witness
coincident or not) when it exists, otherwise the split's own glitches.
With snr_bins > 0 a second figure has one panel per SNR bin (edges = glitch-SNR quantiles).

    python train_utils/compare_efficiency.py [config.yaml | runs/<name>/] [--variants ...] [--split test]
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_curve

from visualize import PALETTE, embed

plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12})

DEADTIMES = (0.01, 0.05, 0.10)


def curve(y, probs, g, sig):
    m = (y == g) | (y == sig)
    deadtime, eff, _ = roc_curve(y[m] == g, probs[m, g])
    pred = probs.argmax(1)
    argmax = ((pred[y == sig] == g).mean(), (pred[y == g] == g).mean())
    at = {f"{int(d * 100)}%": float(eff[deadtime <= d].max()) for d in DEADTIMES}
    return deadtime, eff, argmax, at


def style(ax, max_deadtime):
    ax.set_xlim(0, 100 * max_deadtime)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, out, summary):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out.with_suffix('.pdf')}, .png and .json")


def plot_snr_bins(results, g, sig, n_bins, max_deadtime, out):
    y, snr = results[0]["y"], results[0]["snr"]
    edges = np.quantile(snr[y == g], np.linspace(0, 1, n_bins + 1))
    bins = np.digitize(snr, edges[1:-1])

    fig, axes = plt.subplots(1, n_bins, figsize=(3.2 * n_bins, 3.6), sharey=True)
    summary = {"edges": edges.tolist(), "bins": []}
    for b, ax in enumerate(axes):
        in_bin = bins == b
        n_g, n_s = int((in_bin & (y == g)).sum()), int((in_bin & (y == sig)).sum())
        entry = {"snr_range": [float(edges[b]), float(edges[b + 1])], "n_glitch": n_g, "n_signal": n_s, "models": {}}
        for i, r in enumerate(results):
            if n_g == 0 or n_s == 0:
                continue
            deadtime, eff, argmax, at = curve(r["y"][in_bin], r["probs"][in_bin], g, sig)
            ax.plot(100 * deadtime, eff, lw=2, color=PALETTE[i], drawstyle="steps-post", label=r["label"])
            entry["models"][r["label"]] = {"efficiency_at_deadtime": at}
        summary["bins"].append(entry)
        style(ax, max_deadtime)
        ax.set_title(f"SNR {edges[b]:.1f}–{edges[b + 1]:.1f}", fontsize=12)
        ax.set_xlabel("Deadtime (%)")
    axes[0].set_ylabel("Efficiency")
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, out, summary)


def make_result(label, ckpt, e, label_names, e_glitch=None):
    """A compare() entry from a model's split embeddings e; with e_glitch (embeddings of the test glitch
    population) the split's glitches are swapped for those."""
    y, snr, probs = e["y"], e["snr"], softmax(e["logits"], axis=1)
    if e_glitch is not None:
        keep = y != {v: k for k, v in label_names.items()}["glitch"]
        y = np.concatenate([y[keep], e_glitch["y"]])
        snr = np.concatenate([snr[keep], e_glitch["snr"]])
        probs = np.concatenate([probs[keep], softmax(e_glitch["logits"], axis=1)])
    r = {"label": label, "ckpt": str(ckpt), "y": y, "snr": snr, "probs": probs}
    if e_glitch is not None:
        r["population_probs"] = softmax(e_glitch["logits"], axis=1)
    return r


def population_breakdown(results, label_names, witness_matched, out):
    """For the test glitch population: each model's predicted-class fractions and its efficiency at the
    1/5/10% deadtime thresholds, overall and split by witness-matched / unmatched glitches."""
    ids = {v: k for k, v in label_names.items()}
    g, sig = ids["glitch"], ids["signal"]
    n = len(results[0]["population_probs"])
    groups = {"all": np.ones(n, dtype=bool)}
    if witness_matched is not None:
        groups |= {"witness matched": witness_matched, "unmatched": ~witness_matched}
    summary = {"n": {k: int(m.sum()) for k, m in groups.items()}, "models": {}}
    print("population glitches: " + ", ".join(f"{k} {v}" for k, v in summary["n"].items()))

    for r in results:
        pp, p_sig = r["population_probs"], r["probs"][r["y"] == sig, g]
        pred = pp.argmax(1)
        thr = {f"{int(d * 100)}%": float(np.quantile(p_sig, 1 - d)) for d in DEADTIMES}
        entry = {}
        for k, m in groups.items():
            if not m.any():
                continue
            entry[k] = {"predicted_class": {label_names[c]: float((pred[m] == c).mean()) for c in label_names},
                        "efficiency_at_deadtime": {d: float((pp[m, g] > t).mean()) for d, t in thr.items()}}
            fracs = "  ".join(f"{c} {v:.2f}" for c, v in entry[k]["predicted_class"].items())
            effs = "  ".join(f"{d} {v:.3f}" for d, v in entry[k]["efficiency_at_deadtime"].items())
            print(f"  {r['label']:>16} | {k:>15}: eff {effs} | predicted as: {fracs}")
        summary["models"][r["label"]] = entry
    path = out.with_name(out.stem + "_population_breakdown.json")
    path.write_text(json.dumps(summary, indent=2))
    print(f"saved {path}")
    return summary


def compare(results, label_names, out, max_deadtime=0.10, snr_bins=0, glitch_source="split", witness_matched=None):
    """results: list of {"label", "ckpt", "y", "snr", "probs"} (see make_result) on the same split."""
    ids = {v: k for k, v in label_names.items()}
    g, sig = ids["glitch"], ids["signal"]
    print(f"glitches: {glitch_source} ({(results[0]['y'] == g).sum():,}), "
          f"signals: split ({(results[0]['y'] == sig).sum():,})")

    fig, ax = plt.subplots(figsize=(5, 3.75))
    summary = {"glitch_source": glitch_source, "n_glitch": int((results[0]["y"] == g).sum()),
               "n_signal": int((results[0]["y"] == sig).sum())}
    for i, r in enumerate(results):
        deadtime, eff, argmax, at = curve(r["y"], r["probs"], g, sig)
        summary[r["label"]] = {"ckpt": r["ckpt"], "efficiency_at_deadtime": at,
                               "argmax": {"deadtime": float(argmax[0]), "efficiency": float(argmax[1])}}
        ax.plot(100 * deadtime, eff, lw=2, color=PALETTE[i], drawstyle="steps-post", label=r["label"])
        print(f"{r['label']}: {at}  argmax deadtime {100 * argmax[0]:.2f}%, eff {argmax[1]:.3f}")

    style(ax, max_deadtime)
    ax.set_xlabel("Deadtime (%)")
    ax.set_ylabel("Efficiency")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, out, summary)

    if "population_probs" in results[0]:
        population_breakdown(results, label_names, witness_matched, out)
    if snr_bins:
        plot_snr_bins(results, g, sig, snr_bins, max_deadtime, out.with_name(out.stem + f"_snr{snr_bins}bins.png"))
    return summary


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import experiment_setup as xs

    cfg, args = xs.parse_args("Overlay the efficiency vs deadtime curves of an experiment's trained runs.")
    ev = cfg["evaluation"]
    loaders, meta = xs.get_loaders(cfg)
    glitch_loader = xs.get_test_glitch_loader(cfg)
    results = []
    for variant in args.variants:
        module, ckpt = xs.load_module(cfg, variant, meta)
        e = embed(module, loaders[args.split], xs.DEVICE)
        e_glitch = embed(module, glitch_loader, xs.DEVICE) if glitch_loader else None
        results.append(make_result(variant.replace("_", " "), ckpt, e, meta["label_names"], e_glitch))
    compare(results, meta["label_names"], xs.out_root(cfg) / f"efficiency_deadtime_compare_{args.split}.png",
            ev["max_deadtime"], ev["snr_bins"], "all_test_glitches" if glitch_loader else "split",
            getattr(glitch_loader, "witness_matched", None))


if __name__ == "__main__":
    main()
