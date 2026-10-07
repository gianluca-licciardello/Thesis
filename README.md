# Thesis experiment code

This repository contains runnable model code and setup instructions. Results,
trained checkpoints, plots, logs and datasets are excluded from Git. Input
bundles are shared separately.

| System | Experiments | Instructions |
|---|---|---|
| MDERNet | AIDE Full/Clean/Balanced; PPB-Emo Full/Clean/EEG/EPQ/Cluster, independent and mixed | [MDERNet README](MDERNet/README.md) |
| DECNet | 23 historical configurations: AIDE Full/Balanced/Clean; PPB-Emo five subsets, two modalities and two split conditions | [DECNet README](DECNet/README.md) |
| UbH-GCN | 27 saved experiment groups, four joint/bone streams each, plus ensembles | [UbH-GCN README](UbH-GCN/README.md) |

MDERNet uses the validation-based 50-epoch protocol, with LR 0.01 for AIDE and
0.0001 for PPB-Emo. Run its complete suite with `bash current/run_all.sh` from
`MDERNet` after configuring the environment and input bundle.

DECNet uses outer fold 0 and five inner folds, with validation-based selection.
Its package reruns the historical configurations using available source code;
exact archived numerical reproduction is not certified because original
immutable code/input snapshots have not been established. See its README for
scope, input export and `historical_suite.py` commands.

Each model requires its own input bundle and Python environment. The packages use different dependency requirements and evaluation protocols.

UbH-GCN reproduces saved run configurations with fixed train/eval/test partitions
and ordinary eval-accuracy checkpoint selection. See its README for the
separate prepared input bundle and `run_saved.py` commands.
