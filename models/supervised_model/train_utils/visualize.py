"""Embedding extraction and corner plots of strain_z / witness_z.

    python train_utils/visualize.py [config.yaml | runs/<name>/] [--variants ...] [--split test]
writes runs/<name>/<variant>/plots/ for already trained runs (run_experiment.py does the same after training).
"""

import torch  # before matplotlib: the other order can segfault when lightning is imported later (Windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


@torch.no_grad()
def embed(module, loader, device):
    module.eval().to(device)
    out = {"strain_z": [], "witness_z": [], "logits": [], "y": [], "y_det": [], "special": [], "snr": []}
    for batch in loader:
        x, y, y_det, special = batch[:4]
        if len(batch) > 4:
            out["snr"].append(batch[4])
        o = module(x.to(device))
        for k in ("strain_z", "witness_z", "logits"):
            if k in o:
                out[k].append(o[k].cpu())
        out["y"].append(y)
        out["y_det"].append(y_det)
        out["special"].append(special)
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}

def corner_plot(z, groups, names, title, path, special=None, max_points=4000, seed=0):
    d = z.shape[1]
    rng = np.random.default_rng(seed)
    keys = [k for k in sorted(names) if (groups == k).any()]
    per = max_points // len(keys)
    idx = np.concatenate([rng.permutation(np.flatnonzero(groups == k))[:per] for k in keys])
    rng.shuffle(idx)
    colors = np.array(PALETTE)[groups[idx] % len(PALETTE)]

    fig, axes = plt.subplots(d, d, figsize=(1.8 * d + 1, 1.8 * d + 1))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                bins = np.linspace(z[:, i].min(), z[:, i].max(), 40)
                for k in keys:
                    ax.hist(z[groups == k, i], bins=bins, density=True, histtype="step", lw=1.5, color=PALETTE[k])
                ax.set_yticks([])
            else:
                ax.scatter(z[idx, j], z[idx, i], s=3, alpha=0.35, lw=0, c=colors, rasterized=True)
                if special is not None and special.any():
                    ax.scatter(z[special, j], z[special, i], s=160, marker="*", c="#111111", edgecolors="white", zorder=5)
            ax.tick_params(labelsize=6)
            if i < d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(f"z{j}")
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(f"z{i}")
            ax.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([], [], marker="o", ls="", ms=7, color=PALETTE[k], label=f"{names[k]} (n={(groups == k).sum()})")
               for k in keys]
    if special is not None and special.any():
        handles.append(Line2D([], [], marker="*", ls="", ms=11, color="#111111", label="special event"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.95), frameon=False)
    fig.suptitle(title, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def visualize(e, meta, out_dir, split="test", witness_label="detector", color_by="auto"):
    """color_by="auto": strain_z by class, witness_z by the label it was trained on."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"embeddings_{split}.npz", **e)

    labels = {"class": (e["y"], meta["label_names"]), "detector": (e["y_det"], meta["detector_names"])}
    trained_on = {"strain_z": "class", "witness_z": witness_label}
    for key in ("strain_z", "witness_z"):
        by = trained_on[key] if color_by == "auto" else color_by
        groups, names = labels[by]
        corner_plot(e[key], groups, names, f"{key} - {split}, colored by {by}",
                    out_dir / f"corner_{key}_{split}_{by}.png", special=e["special"].astype(bool))


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import experiment_setup as xs

    cfg, args = xs.parse_args("Corner plots of strain_z / witness_z for an experiment's trained runs.")
    loaders, meta = xs.get_loaders(cfg)
    for variant in args.variants:
        module, _ = xs.load_module(cfg, variant, meta)
        visualize(embed(module, loaders[args.split], xs.DEVICE), meta, xs.out_root(cfg) / variant / "plots",
                  args.split, module.hparams.witness_label)


if __name__ == "__main__":
    main()
