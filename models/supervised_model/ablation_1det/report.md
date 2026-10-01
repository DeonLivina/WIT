# Witness ablation: downsample model (1 seed(s): 42)

Test split. eff@X% = fraction of glitches flagged as glitch when X% of signal injections are flagged (deadtime). blip->glitch = blips predicted as glitch.

| metric | with witness | no witness | with witness, zeroed at test | delta (with - no witness) |
|---|---|---|---|---|
| accuracy | 0.875 | 0.854 | 0.693 | 0.021 |
| balanced_accuracy | 0.864 | 0.835 | 0.637 | 0.029 |
| macro_f1 | 0.866 | 0.828 | 0.625 | 0.038 |
| roc_auc_macro_ovr | 0.951 | 0.959 | 0.859 | -0.008 |
| eff@1% | 0.736 | 0.486 | 0.292 | 0.250 |
| eff@5% | 0.802 | 0.612 | 0.306 | 0.190 |
| eff@10% | 0.834 | 0.720 | 0.312 | 0.114 |
| blip->glitch | 0.007 | 0.025 | 0.002 | -0.019 |
| glitch->blip | 0.018 | 0.040 | 0.198 | -0.022 |

## Per-class recall

| class | with witness | no witness | with witness, zeroed at test |
|---|---|---|---|
| background | 0.887 | 0.870 | 0.404 |
| glitch | 0.788 | 0.708 | 0.238 |
| signal | 0.838 | 0.824 | 0.962 |
| blip | 0.942 | 0.936 | 0.945 |

## Per seed

- seed 42: with witness: acc 0.875, eff@5% 0.802; no witness: acc 0.854, eff@5% 0.612; with witness, zeroed at test: acc 0.693, eff@5% 0.306
