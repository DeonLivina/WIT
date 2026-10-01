"""Evaluation of one trained model from its embeddings (see visualize.embed).

Classifier: confusion matrix, one-vs-rest ROC, accuracy and imbalance-aware scores.
Witness space: linear probe witness_z -> H1/L1, fit on train, scored on the eval split.
Efficiency vs deadtime: threshold sweep on p(glitch).

    python train_utils/evaluate.py [config.yaml | runs/<name>/] [--variants ...] [--split test]
writes runs/<name>/<variant>/eval/ for already trained runs (run_experiment.py does the same after training).
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)

from visualize import PALETTE, embed


def compute_metrics(y, probs, names):
    pred = probs.argmax(1)
    labels = list(range(len(names)))
    return {
        "n": int(len(y)),
        "class_counts": {names[k]: int((y == k).sum()) for k in labels},
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "roc_auc_macro_ovr": roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=labels),
        "roc_auc_weighted_ovr": roc_auc_score(y, probs, multi_class="ovr", average="weighted", labels=labels),
        "per_class": classification_report(y, pred, labels=labels, target_names=[names[k] for k in labels],
                                           output_dict=True, zero_division=0),
    }


def plot_confusion(y, pred, names, path):
    labels = list(range(len(names)))
    cm = confusion_matrix(y, pred, labels=labels)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ticks = [names[k] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in ((axes[0], cm, "counts"), (axes[1], norm, "row-normalized (recall)")):
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in labels:
            for j in labels:
                text = f"{mat[i, j]:,}" if mat is cm else f"{mat[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="white" if norm[i, j] > 0.5 else "#222222")
        ax.set_xticks(labels, ticks, rotation=30, ha="right")
        ax.set_yticks(labels, ticks)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_roc(y, probs, names, macro_auc, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for k in range(len(names)):
        if not (y == k).any():
            continue
        fpr, tpr, _ = roc_curve(y == k, probs[:, k])
        ax.plot(fpr, tpr, lw=2, color=PALETTE[k], label=f"{names[k]} vs rest (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999999")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"One-vs-rest ROC (macro AUC {macro_auc:.3f})", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def efficiency_deadtime(y, probs, names, path, max_deadtime=0.10):
    """Threshold sweep on p(glitch). Efficiency = fraction of true glitches flagged as glitch,
    deadtime = fraction of signal injections flagged as glitch."""
    ids = {v: k for k, v in names.items()}
    g, sig = ids["glitch"], ids["signal"]
    m = (y == g) | (y == sig)
    deadtime, eff, _ = roc_curve(y[m] == g, probs[m, g])

    pred = probs.argmax(1)
    argmax_point = ((pred[y == sig] == g).mean(), (pred[y == g] == g).mean())
    at = {f"{int(d * 100)}%": float(eff[deadtime <= d].max()) for d in (0.01, 0.05, 0.10)}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(100 * deadtime, eff, lw=2, color=PALETTE[g], drawstyle="steps-post")
    ax.plot(100 * argmax_point[0], argmax_point[1], "o", ms=9, color="#222222",
            label=f"argmax decision ({100 * argmax_point[0]:.2f}% deadtime, {argmax_point[1]:.3f} eff.)")
    for d, e in zip((1, 5, 10), at.values()):
        ax.annotate(f"{e:.3f}", (d, e), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
        ax.plot(d, e, "o", ms=5, color=PALETTE[g])
    ax.set_xlim(0, 100 * max_deadtime)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("deadtime: signal injections flagged as glitch (%)")
    ax.set_ylabel("efficiency: true glitches flagged as glitch")
    ax.set_title(f"Glitch efficiency vs deadtime ({(y == g).sum():,} glitches, {(y == sig).sum():,} signals)", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")
    return {"efficiency_at_deadtime": at,
            "argmax": {"deadtime": float(argmax_point[0]), "efficiency": float(argmax_point[1])}}


def witness_detector_eval(train_e, e, class_names, det_names, out_dir, split):
    """Linear probe witness_z -> detector: how well H1 and L1 separate in witness space."""
    probe = LogisticRegression(max_iter=1000).fit(train_e["witness_z"], train_e["y_det"])
    score = probe.predict_proba(e["witness_z"])[:, 1]
    pred, y_det = (score > 0.5).astype(int), e["y_det"]

    per_class = {}
    for k, name in class_names.items():
        m = e["y"] == k
        if len(np.unique(y_det[m])) == 2:
            per_class[name] = {"roc_auc": roc_auc_score(y_det[m], score[m]),
                               "balanced_accuracy": balanced_accuracy_score(y_det[m], pred[m]), "n": int(m.sum())}
    metrics = {
        "accuracy": accuracy_score(y_det, pred),
        "balanced_accuracy": balanced_accuracy_score(y_det, pred),
        "roc_auc": roc_auc_score(y_det, score),
        "per_class": per_class,
    }

    plot_confusion(y_det, pred, det_names, out_dir / f"witness_detector_confusion_{split}.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y_det, score)
    ax.plot(fpr, tpr, lw=3, color="#222222", label=f"all (AUC {metrics['roc_auc']:.3f})")
    for k, name in class_names.items():
        m = e["y"] == k
        if name in per_class:
            fpr, tpr, _ = roc_curve(y_det[m], score[m])
            ax.plot(fpr, tpr, lw=1.5, color=PALETTE[k], label=f"{name} (AUC {per_class[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999999")
    ax.set_xlabel(f"false positive rate ({det_names[0]} called {det_names[1]})")
    ax.set_ylabel("true positive rate")
    ax.set_title("witness_z: H1 vs L1 linear probe", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"witness_detector_roc_{split}.png", dpi=150)
    plt.close(fig)
    print(f"saved {out_dir / f'witness_detector_roc_{split}.png'}")
    return metrics


def evaluate(e, train_e, meta, out_dir, split="test"):
    """Writes plots and metrics_<split>.json to out_dir; returns the metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y, probs, names = e["y"], softmax(e["logits"], axis=1), meta["label_names"]

    metrics = compute_metrics(y, probs, names)
    metrics["efficiency_deadtime"] = efficiency_deadtime(y, probs, names, out_dir / f"efficiency_deadtime_{split}.png")
    metrics["witness_detector_probe"] = witness_detector_eval(train_e, e, names, meta["detector_names"], out_dir, split)
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(classification_report(y, probs.argmax(1), labels=list(names), target_names=list(names.values()), zero_division=0))

    plot_confusion(y, probs.argmax(1), names, out_dir / f"confusion_matrix_{split}.png")
    plot_roc(y, probs, names, metrics["roc_auc_macro_ovr"], out_dir / f"roc_curves_{split}.png")
    return metrics


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import experiment_setup as xs

    cfg, args = xs.parse_args("Evaluate an experiment's trained runs.")
    loaders, meta = xs.get_loaders(cfg)
    for variant in args.variants:
        print(f"\n=== {cfg['experiment']['model']} / {variant} ===")
        module, _ = xs.load_module(cfg, variant, meta)
        evaluate(embed(module, loaders[args.split], xs.DEVICE), embed(module, loaders["train"], xs.DEVICE),
                 meta, xs.out_root(cfg) / variant / "eval", args.split)


if __name__ == "__main__":
    main()
