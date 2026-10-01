#!/usr/bin/env python3
"""Run the full data generation (triggers -> extraction -> injections -> stats) for each detector,
saving everything under auto_data/<DET>/.  e.g. python run_pipeline.py --detectors H1 L1 --steps all"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent

# In dependency order; each step reads what the earlier ones wrote for the same detector.
steps = {
    "bg_triggers": "bg_triggers.py",
    "coincidence": "strain_witness_coincidence.py",
    "background": "extract_background.py",
    "leakage": "check_background_witness_leakage.py",
    "glitches": "extract_glitches.py",
    "long_glitches": "extract_long_glitches.py",
    "test_glitches": "extract_test_glitches.py",
    "signal": "inject_signal.py",
    "blip": "inject_gaussians.py",
}


def run(script, detector, out_root):
    env = dict(os.environ, GW_DETECTOR=detector, GW_OUT_ROOT=str(out_root))
    print(f"\n=== {detector}: {script} ===", flush=True)
    t0 = time.time()
    if subprocess.run([sys.executable, script], cwd=here, env=env).returncode != 0:
        sys.exit(f"\nStopped: {script} failed for {detector} (see error above). "
                 f"Rerun from this step with --steps.")
    print(f"=== {detector}: {script} done in {time.time() - t0:.0f}s ===", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detectors", nargs="+", default=["H1", "L1"], choices=["H1", "L1"])
    p.add_argument("--steps", nargs="+", default=["all"], choices=["all"] + list(steps),
                   help="subset of steps to run, always executed in pipeline order")
    p.add_argument("--out", default=str(here.parent / "auto_data"), help="output root folder")
    p.add_argument("--no_stats", action="store_true", help="skip dataset_stats.py at the end")
    args = p.parse_args()

    selected = list(steps) if "all" in args.steps else [s for s in steps if s in args.steps]
    out_root = Path(args.out).resolve()
    print(f"Output: {out_root}\nDetectors: {args.detectors}\nSteps: {selected}")

    for det in args.detectors:
        for name in selected:
            run(steps[name], det, out_root)
    if not args.no_stats:
        run("dataset_stats.py", args.detectors[0], out_root)

    print(f"\nAll done. Data in {out_root}")


if __name__ == "__main__":
    main()
