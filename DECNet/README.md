# DECNet historical experiment configurations

This package runs the 23 experiment configurations described by the local
`DECNet/RESULTS_HISTORY.md` archive. Results, checkpoints, plots, logs, datasets
and the historical score table are deliberately not included in Git.

## Reproduction scope

The historical report is explicitly marked obsolete in the source workspace.
The available training, evaluation and preprocessing code was subsequently
modified. No immutable original source/input snapshot has been established for
all archived rows. Therefore this package reruns the historical **experiment
configurations using the available implementation**; it does not certify exact
reproduction of the archived numerical scores. Do not label newly generated
scores as the original historical results. Exact numerical reproduction would
require the original code/environment and unmodified input splits/features.

The corrected 100-epoch suite mentioned at the end of the history document is
a separate later experiment and is not included in this 50-epoch launcher.
Paper-reference rows are published comparisons, not local runs to reproduce.

## Experiments and evaluation

- AIDE Full, Balanced and Clean Keypoints: 45-channel FastSAM3D upper-body input,
  five classes, with no enforced participant separation.
- PPB-Emo Full, Cluster, EPQ, EEG and Clean Keypoints: both participant-independent
  and mixed conditions, each using either 8-channel CAN-bus input or 45-channel
  FastSAM3D input. This gives 20 PPB experiments.

Each experiment trains a dual-branch `V-DB` model and reports its facial,
secondary-modality and fused predictions. These outputs are not three
independently trained single-modality models. The launcher uses outer fold 0
and five inner folds, selects the checkpoint/model through validation balanced
accuracy and evaluates the selected inner-fold model on the fixed test split.
This is not the ten-fold 8/1/1 protocol used in the MDERNet package.

Settings follow the available historical shell launcher/defaults: 50 epochs,
batch size 32, three-second windows, SGD LR 0.01, momentum 0.9, weight decay
0.0001, DB noise amplitude 0.1, seed 42 and eight loader workers. Architecture
defaults are spatial depth 1, temporal depth 3, nf=32 and Inception depth 6.
Training uses StepLR(step_size=100, gamma=0.1), which does not decay LR during
these 50 epochs. The later optional training-fold normalization and mild face
crop switches are not enabled. Annotation sample weights are preserved.
No external pretrained model is loaded by the default training path.

PPB class order is AD, DD, FD, HD, ND, SAD, SD. AIDE class order is Anger,
Anxiety, Happiness, Peace, Weariness. This AIDE ordering differs from MDERNet.

## Environment

Use Linux, Python 3.10 and a CUDA GPU. From `Thesis/DECNet`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a separate environment from MDERNet because the pinned PyTorch versions
differ. The included requirements match the available DECNet runtime stack.

## Required external inputs

Share a DECNet input bundle separately:

```text
DECNet-share-inputs/
  annotation/*.txt
  frames/<directory-id>/*.jpg
  data/aide_db_features.pkl
  data/db_features_3s_stride3s.pkl
  data/ppb_kp_features_3s_stride3s.pkl
  INPUTS.json
```

These are DECNet-specific windowed face sequences, feature dictionaries and
outer/inner split annotations. The MDERNet input bundle cannot replace them.
The annotation columns are frame-directory, frame count, class index,
feature-dictionary key and optional sample weight. Preserve feature keys,
class order, weights and split membership. Subset membership is already
encoded in the prepared annotations; the launcher does not regenerate it
from the newer subset JSONs. The historical report's subset counts and
interpretations were superseded, so those prose descriptions are not used
to recreate selection rules.

The exporter reads only prepared inputs and copies their frame directories;
it never exports trained models or results. It rewrites absolute frame paths
to portable bundle-relative paths. Input validation checks annotation files,
feature-key/channel coverage and frame availability. Pickles must come from
a trusted source, such as your own prepared data.

On the original machine, from this repository's DECNet directory:

```bash
python export_inputs.py --source-root /data/gianluca/scripts/DECNet --check-only
python export_inputs.py \
    --source-root /data/gianluca/scripts/DECNet \
    --destination /data/gianluca/DECNet-share-inputs
```

The destination must not exist. This copies all required frames and can use
substantial disk space. Share the resulting directory separately from GitHub.
No raw EEG recordings or questionnaire files are needed when these prepared
annotations/features are provided. Optional label-group reports are omitted;
the main emotion-class evaluation does not require `label_groups.json`.

## Run

```bash
export DECNET_INPUTS=/path/to/DECNet-share-inputs
export CUDA_VISIBLE_DEVICES=0
python historical_suite.py --data-root "$DECNET_INPUTS" --check-only
python -u historical_suite.py --data-root "$DECNET_INPUTS"
```

Run one configuration with `--only`, for example:

```bash
python -u historical_suite.py --data-root "$DECNET_INPUTS" \
    --only ppb_full_fast_sam3d_independent
```

Use `python historical_suite.py --help` for all 23 names. Experiments run
sequentially and stop on an error. Results are created locally under `results/`
(or `--results-dir`), with inner-fold checkpoints/logs and held-out evaluation.
Prepared absolute-path annotations are written inside that output directory.
The underlying runner uses timestamped directories for existing experiments;
this launcher does not promise automatic training resume.

## Included code

`historical_suite.py` defines and launches the 23 configurations;
`run_experiment.py` coordinates inner-fold training and final evaluation;
`train.py`, `test.py`, `models/` and `dataloader/` implement the runtime.
`export_inputs.py` packages existing inputs without regenerating splits or
rerunning pose estimation. No source paper, historical results or generated
figures are part of this package.
