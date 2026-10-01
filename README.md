# GWfinal

Classify 1 s LIGO strain windows (H1 / L1, O3a) as **background, glitch, signal or blip**, using the strain
together with auxiliary witness channels, and measure what the witness channels add (with vs. without witness).

## Layout

```
data_pipeline/            build the datasets: triggers -> whitened HDF5 files in auto_data/<DET>/
dataset/                  waveform / injection helpers used by the signal and blip injections
models/supervised_model/
  config.yaml             experiment settings (model, data, training, evaluation)
  run_experiment.py       train both variants, evaluate, plot, compare efficiency vs deadtime
  experiment_setup.py     shared config / data / checkpoint helpers
  main_model/             main model (separate strain + witness encoders, Mamba)
  downsample_model/       downsample model
  train_utils/            loader, losses, evaluate, visualize, compare_efficiency
```

## Setup

```
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match your CUDA version
pip install -r requirements.txt
```

`mamba-ssm` needs Linux and an NVIDIA GPU.

## Data

Raw data, trigger files and the generated HDF5 datasets: **Google Drive: <link>**

Place them relative to the repo root:

| What | Where |
|---|---|
| strain (GWOSC HDF5) | `data/og/` |
| witness channels | `data/witness_data/` |
| Omicron triggers | `triggers_H1/`, `triggers_L1/` |
| generated datasets (`whitened_*.h5`, `all_test_glitches.h5`, CSVs) | `auto_data/H1/`, `auto_data/L1/` |

If you download the generated datasets you can skip the pipeline and go straight to training.
Set `data.data_dir` in `models/supervised_model/config.yaml` to your `auto_data` folder (`null` = `GWfinal/auto_data`).

## Build the datasets

```
cd data_pipeline
python run_pipeline.py --detectors H1 L1 --steps all       # or a subset, e.g. --steps test_glitches
```

Steps, in order: `bg_triggers, coincidence, background, leakage, glitches, long_glitches, test_glitches,
signal, blip`, then `dataset_stats.py`. Output goes to `auto_data/<DET>/` (`--out` to change it).
`test_glitches` extracts every SNR >= 7 strain trigger after the test cutoff (`all_test_glitches.h5`),
used as the glitch population for the efficiency vs deadtime curves.

## Train and evaluate

```
cd models/supervised_model
python run_experiment.py [config.yaml]
```

Trains the model in `experiment.model` (`main` or `downsample`) with and without witness channels, then writes
evaluation, embedding plots and the efficiency vs deadtime comparison to `runs/<experiment.name>/`.
Existing checkpoints are reused unless `experiment.retrain: true` or the config changed.

Each step also runs on its own for already trained runs:

```
python train_utils/evaluate.py           [config.yaml | runs/<name>/] [--variants ...] [--split test]
python train_utils/visualize.py          [config.yaml | runs/<name>/] [--variants ...] [--split test]
python train_utils/compare_efficiency.py [config.yaml | runs/<name>/] [--variants ...] [--split test]
```

The test cutoff (`data.test_cutoff_utc`, default 2019-05-05 00:00 UTC) must match `test_cutoff_utc` in
`data_pipeline/config.py`.
