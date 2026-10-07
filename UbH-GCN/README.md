# UbH-GCN saved experiment suite

This package reruns the 108 training configurations saved in the original
`UbH-GCN/work_dir`, then generates the corresponding ensembles and reports.
Only source code, run configurations and configuration-provenance hashes are
included. Results, logs, trained weights and input arrays are excluded from Git.
The upstream license is retained in `LICENSE`.

## Coverage

There are 27 experiment groups, each with four streams: `joint_root_1`,
`bone_root_1`, `joint_root_14` and `bone_root_14`.

- AIDE original AlphaPose annotations: one group.
- AIDE FastSAM3D: 2D and 3D, each Full, Balanced and Clean Keypoints: six groups.
- PPB-Emo FastSAM3D: 2D and 3D, independent and mixed splits, each Full,
  Cluster, EPQ, EEG and Clean Keypoints: twenty groups.

Every group produces two joint+bone ensembles (one for each root) and a
four-stream ensemble. Scores are L2-normalized and summed by `ensemble.py`.
The original `report.py` generates the per-group table and training plots.

The YAML files in `configs/` were extracted from the saved run configurations,
not inferred from historical scores. `configuration_provenance.json` records
hashes of the original saved YAML and portable copies. Only configuration
references and output paths were rewritten during extraction. The launcher
resolves input paths and overrides the GPU/output location at runtime.
The current model source matches the model snapshots found in the saved runs.
A complete immutable snapshot of every historical dependency/feeder/input is
not established, so exact numerical equality with archived scores is not
promised. No historical outcomes are bundled as reproduction targets.

## Settings and evaluation

All saved configurations use 90 epochs, seed 1, batch size 64, initial LR 0.1,
SGD with Nesterov, weight decay 0.0004, dropout 0.25 and BlvLoss. Class counts,
input channels, root, bone mode and feeder options are preserved per YAML.
The implementation uses five warm-up epochs followed by cosine decay to
`base_lr * lr_ratio` (0.0001). The YAML's `step` list does not replace that
cosine schedule in the available implementation.

The prepared arrays contain fixed train/eval/test partitions. The trainer
selects the best model using **ordinary eval accuracy**, with the earliest
strictly best checkpoint retained, then reloads it for test evaluation. This
is not MDERNet's balanced-accuracy selection or its rotating ten-fold 8/1/1
protocol. Independent PPB configurations use participant-separated prepared
partitions; mixed configurations allow participant overlap. Subset membership
and split assignments are preserved through the supplied prepared arrays.

The scripts report ordinary accuracy, weighted F1, balanced accuracy and macro
F1 where supported by the individual/ensemble reports. Do not confuse these
metrics or assume the protocols match those of the other model packages.

## Environment

Use Linux, Python 3.10 and a CUDA GPU. From `Thesis/UbH-GCN`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --help
```

The requirements install the local `torchlight` package in editable mode.
Run installation from this directory so its relative path resolves correctly.
Use a separate environment from the other models. The packaged torchlight
code treats its optional Pavi logging import as optional; Pavi is not needed.
No pretrained weights are required: all saved configs start with `weights: null`.

## External input bundle

The prepared input bundle is about 175 MB and contains 162 unique files:
train/eval/test `.npy` arrays and corresponding label `.pkl` files, preserving
the original `data/AIDE/...` and `data/PPB_Emo/...` layout, plus `manifest.json`
with SHA-256 hashes. Only trusted label pickles should be loaded.

Export from the original machine:

```bash
python export_inputs.py \
    --source-root /data/gianluca/scripts/UbH-GCN \
    --destination /data/gianluca/UbH-GCN-share-inputs
```

The destination must be new. That folder has already been prepared in the
original workspace; do not run the export again to the same destination.
Share it separately from GitHub. With this bundle, raw videos, raw pose
estimates, EEG/questionnaire files and external pretrained checkpoints are
not needed. MDERNet and DECNet input bundles are not interchangeable with it.
The exporter packages existing preprocessing outputs rather than rerunning
preprocessing or changing archived split membership.

## Validate and run

After installing dependencies, set the bundle location:

```bash
export CUDA_VISIBLE_DEVICES=0
export UBH_INPUTS=/path/to/UbH-GCN-share-inputs
python run_saved.py --data-root "$UBH_INPUTS" --check-only
python -u run_saved.py --data-root "$UBH_INPUTS" --output-root ./work_dir/reproduction
```

This runs all 108 fits sequentially, followed by ensembles and reports for each
group. It stops on an error. `--check-only` verifies hashes, array dimensions,
label lengths/ranges and the four-stream configurations without training.

To run only one group:

```bash
python -u run_saved.py --data-root "$UBH_INPUTS" \
    --group ppb_emo_fast_sam3d_independent \
    --output-root ./work_dir/ppb_reproduction
```

`python run_saved.py --help` lists all group names. `--device 0` refers to the
first GPU visible to the process. Select an unoccupied GPU for training.
Existing group output directories are rejected to avoid overwriting trained
models; choose a fresh output root for a rerun. Automatic training resume is
not implemented by this wrapper.

## Generated artifacts

Each output group contains four stream folders with runtime configuration,
checkpoints, training/eval histories, test predictions, confusion matrices
and summaries, followed by group ensemble figures/reports. All are generated
locally under the selected output root. No original `work_dir` result files
are included in Git; only their configuration settings were extracted.
