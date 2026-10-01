"""Shared by run_experiment.py and the standalone train_utils scripts (evaluate.py, visualize.py,
compare_efficiency.py): the config, data loaders and checkpoints of an experiment's runs."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _d in [HERE / d for d in ("main_model", "downsample_model", "train_utils", ".")] + [HERE.parent / "common"]:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import torch
import yaml

import loader
from train import SupervisedModule
from train_downsample import DownsampleModule

MODELS = {"main": SupervisedModule, "downsample": DownsampleModule}
VARIANTS = {"with_witness": False, "without_witness": True}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")


def load_config(path):
    """path: a config.yaml, or a run folder runs/<name>/ holding the config.yaml run_experiment.py saved there."""
    path = Path(path)
    return yaml.safe_load((path / "config.yaml" if path.is_dir() else path).read_text())


def out_root(cfg):
    return HERE / "runs" / cfg["experiment"]["name"]


def test_cutoff_gps(cfg):
    return loader.utc_to_gps(*datetime.fromisoformat(str(cfg["data"]["test_cutoff_utc"])).timetuple()[:6])


def get_loaders(cfg):
    d = cfg["data"]
    if d["data_dir"]:
        loader.DATA_DIR = Path(d["data_dir"])
    train, val, test, meta = loader.get_dataloaders(
        cfg["training"]["batch_size"], cfg["experiment"]["seed"], with_snr=True,
        one_detector_only=d["one_detector_only"],
        train_frac=d["train_frac"], val_frac=d["val_frac"], test_cutoff_gps=test_cutoff_gps(cfg))
    return {"train": train, "val": val, "test": test}, meta


def get_test_glitch_loader(cfg):
    """The test-period glitch population used for efficiency vs deadtime; None if disabled or not extracted."""
    if not cfg["evaluation"].get("population_glitches", True):
        return None
    if cfg["data"]["data_dir"]:
        loader.DATA_DIR = Path(cfg["data"]["data_dir"])
    tg = loader.get_test_glitch_loader(cfg["training"]["batch_size"], cfg["data"]["one_detector_only"],
                                       test_cutoff_gps(cfg))
    if tg is None:
        print(f"no {loader.TEST_GLITCH_FILE} found (run data_pipeline/extract_test_glitches.py): "
              "efficiency uses the split's own glitches")
    return tg


def training_setup(cfg):
    """The config entries a checkpoint depends on."""
    t = {k: v for k, v in cfg["training"].items() if k != "wandb"}
    return {"model_type": cfg["experiment"]["model"], "seed": cfg["experiment"]["seed"], "data": cfg["data"],
            "training": t, "model": cfg["model"], "loss": cfg["loss"]}


def mismatches(module, cfg, variant, meta):
    """Why a trained run doesn't match the current config (empty list if it does)."""
    hp = module.hparams
    reasons = [f"{k}: checkpoint {hp.get(k)} vs config {v}"
               for k, v in (("n_witness", meta["n_witness"]), ("no_witness", VARIANTS[variant])) if hp.get(k) != v]
    summary = out_root(cfg) / variant / "summary.json"
    saved = json.loads(summary.read_text()).get("config") if summary.exists() else None
    if saved is not None:
        saved["data"].pop("long_glitches", None)  # option removed; runs trained before still match
    if saved is not None and saved != json.loads(json.dumps(training_setup(cfg))):
        reasons.append("data/training/model/loss settings changed since it was trained")
    return reasons


def load_module(cfg, variant, meta):
    """The trained module of runs/<name>/<variant>/ and its checkpoint path."""
    ckpt = out_root(cfg) / variant / "checkpoints" / "best.ckpt"
    if not ckpt.exists():
        sys.exit(f"No checkpoint {ckpt}: train it first with run_experiment.py")
    try:
        module = MODELS[cfg["experiment"]["model"]].load_from_checkpoint(ckpt, map_location=DEVICE)
    except RuntimeError as e:
        sys.exit(f"{ckpt} doesn't fit experiment.model={cfg['experiment']['model']} / the config's witness "
                 f"channels (was it trained with another config?):\n{e}")
    reasons = mismatches(module, cfg, variant, meta)
    if reasons:
        print(f"WARNING: {ckpt} doesn't match the config (splits may differ from training):\n  " + "\n  ".join(reasons))
    return module, ckpt


def parse_args(description):
    """Command line of the standalone scripts: (config, args with .variants and .split filled in)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("run", nargs="?", default=str(HERE / "config.yaml"),
                   help="config.yaml, or a run folder runs/<name>/ (default: supervised_model/config.yaml)")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), help="default: experiment.variants")
    p.add_argument("--split", choices=["train", "val", "test"], help="default: evaluation.split")
    args = p.parse_args()
    cfg = load_config(args.run)
    args.variants = args.variants or cfg["experiment"]["variants"]
    args.split = args.split or cfg["evaluation"]["split"]
    return cfg, args
