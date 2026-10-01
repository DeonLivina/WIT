"""Witness ablation: train the configured model with and without witness channels, evaluate and plot each run,
then overlay their efficiency vs deadtime curves.

    python models/supervised_model/run_experiment.py [config.yaml]
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lightning as L
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from experiment_setup import (DEVICE, MODELS, VARIANTS, get_loaders, get_test_glitch_loader, load_config,
                              mismatches, out_root, training_setup)
from compare_efficiency import compare, make_result
from evaluate import evaluate
from visualize import embed, visualize


def train(cfg, variant, loaders, meta, run_dir):
    module_cls, no_witness = MODELS[cfg["experiment"]["model"]], VARIANTS[variant]
    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    if ckpt_path.exists() and not cfg["experiment"]["retrain"]:
        module = module_cls.load_from_checkpoint(ckpt_path, map_location=DEVICE)
        reasons = mismatches(module, cfg, variant, meta)
        if not reasons:
            print(f"reusing {ckpt_path}")
            return module, ckpt_path
        print(f"retraining, {ckpt_path} doesn't match the config:\n  " + "\n  ".join(reasons))
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)

    t = cfg["training"]
    L.seed_everything(cfg["experiment"]["seed"], workers=True)
    module = module_cls(meta["n_witness"], meta["num_classes"], lr=t["lr"], no_witness=no_witness,
                        **cfg["model"], **cfg["loss"])
    ckpt = ModelCheckpoint(run_dir / "checkpoints", filename="best", monitor="val_acc", mode="max")
    name = f"{cfg['experiment']['name']}/{run_dir.name}"
    logger = WandbLogger(project="ligo-supervised-model", name=name) if t["wandb"] else CSVLogger(run_dir, name="logs")
    trainer = L.Trainer(max_epochs=t["epochs"], accelerator="auto", devices=1, logger=logger,
                        callbacks=[ckpt, EarlyStopping("val_acc", mode="max", patience=t["patience"])],
                        gradient_clip_val=t["gradient_clip"])
    trainer.fit(module, loaders["train"], loaders["val"])
    test = trainer.test(module, loaders["test"], ckpt_path="best")[0]

    summary = {"best_ckpt": ckpt.best_model_path, "best_val_acc": float(ckpt.best_model_score), "test": test,
               "config": training_setup(cfg)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return module_cls.load_from_checkpoint(ckpt.best_model_path, map_location=DEVICE), Path(ckpt.best_model_path)


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "config.yaml"
    cfg = load_config(cfg_path)
    exp, ev = cfg["experiment"], cfg["evaluation"]
    root = out_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    loaders, meta = get_loaders(cfg)
    glitch_loader = get_test_glitch_loader(cfg)
    split = ev["split"]
    results = []
    for variant in exp["variants"]:
        print(f"\n=== {exp['model']} / {variant} ===")
        run_dir = root / variant
        module, ckpt = train(cfg, variant, loaders, meta, run_dir)

        e = embed(module, loaders[split], DEVICE)
        evaluate(e, embed(module, loaders["train"], DEVICE), meta, run_dir / "eval", split)
        visualize(e, meta, run_dir / "plots", split, module.hparams.witness_label)
        e_glitch = embed(module, glitch_loader, DEVICE) if glitch_loader else None
        results.append(make_result(variant.replace("_", " "), ckpt, e, meta["label_names"], e_glitch))

    compare(results, meta["label_names"], root / f"efficiency_deadtime_compare_{split}.png",
            ev["max_deadtime"], ev["snr_bins"], "all_test_glitches" if glitch_loader else "split",
            getattr(glitch_loader, "witness_matched", None))
    print(f"\nAll outputs in {root}")


if __name__ == "__main__":
    main()
