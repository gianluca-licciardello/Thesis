# MDERNet: current AIDE and PPB-Emo experiments

Use `current/run_subsets.py` for AIDE Clean Keypoints, AIDE Balanced, PPB-Emo
Full, Clean Keypoints, EEG, EPQ, and Cluster, plus AIDE Full. It packages the training/model code from the active
September 30 campaign and the configuration-first continuation. The older
root-level `ablation_aide.py` is retained for historical use and does **not**
implement this evaluation protocol.

## Environment

Linux, Python 3.10 or 3.11, and a CUDA-capable NVIDIA GPU are required for
training. Input validation can run on CPU. From `Thesis/MDERNet`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Exact settings

| Setting | Value |
|---|---|
| Epochs / seed / dropout | 50 / 42 for every fit / 0.0 |
| Batch size | 64 |
| Optimizer | SGD, momentum 0.9, Nesterov, weight decay 0.0001 |
| Initial learning rate | AIDE: 0.01; PPB-Emo: 0.0001 (independent and mixed) |
| Scheduler | CosineAnnealingWarmRestarts, T_0=5, eta_min=0.00001 |
| Backbone | Dominik ResNet-18; checked weight mapping; fully fine-tuned |
| Input | 30 grayscale 112×112 faces; 15 body joints; xyz, original visibility, seven validity-masked bones |
| Cross-validation | 10 folds: test i, validation (i+1)%10, remaining 8 train |
| Checkpoint selection | Validation macro-accuracy, then macro F1; earliest exact tie |
| AIDE loss | Cross entropy + soft macro F1 |
| PPB-Emo loss | Cross entropy + 0.1 MSE + CCC |

AIDE uses stratified clip folds. PPB-Emo supports two evaluation conditions:

- **Subject-independent** (`--split independent`, the default): participants
  are disjoint across training, validation, and test sets.
- **Mixed** (`--split mixed`): stratified clip folds allow clips from the same
  participant to appear in training, validation, and test sets.

Both conditions use the same ten-fold 8/1/1 protocol and training settings.
Each PPB subset must be run once per condition to evaluate both. AIDE has no
participant-independent option; `--split` only affects PPB-Emo.
Subset membership filters **training only**; validation and test keep their
full held-out folds. EEG/EPQ/Cluster are clip-selection definitions, not
additional EEG or questionnaire model inputs.

Each of the seven configurations runs all ten folds before the next configuration:
FEB, FEB without FAM, FEB without FAM/FM, BGB without refinement, BGB without
refinement/visibility, fusion with refinement, fusion without refinement.
Then one FEB+BGB architecture is selected by mean validation macro-accuracy
(and macro F1 tie-break) across ten folds and evaluated on all ten folds.
An identical already-trained fusion is reused. Fixed-pair scores are
post-selection estimates, not unbiased nested-CV estimates.

**EPQ exception:** independent EPQ training folds contain 46–58 clips. The
launcher keeps the smaller batch when a fold has fewer than 64 clips; retaining
`drop_last=True` would otherwise produce no training batches. Other loader
settings match the campaign. No training settings are inherited from the old
root-level runner.

## External files to share

The Git repository contains code only. For exact reproduction, share this input
bundle separately (approximately 4.6 GiB):

```text
inputs/
  resnet18_dominik.pth
  aide_clean_keypoints_subset.json
  aide_balanced_subset.json
  subject_subsets.json
  ppb_emo_clean_keypoints_subset.json
  aide_body.npy
  ppb_body.npy
  aide/labels.csv
  aide/faces/*.npy
  ppb/labels.csv
  ppb/faces/*.npy
  data_manifest.json
```

The bundled body arrays have shape `(N,30,15,4)` and follow the corresponding
CSV row order exactly. They preserve the campaign's interpolation,
normalization, and original visibility masks. Do not reorder the CSV rows.
AIDE labels contain `clip_id,discrete_label`; PPB labels include `participant`,
`emotion_code`, `discrete_label`, `valence`, `arousal`, and `dominance`.

The required backbone SHA-256 is
`734341508e3ddbcd181e40da0173cf2c77cd0de3bc2455715cc1b7f42a63a6f6`.
ImageNet weights are not an equivalent replacement for these experiments.

The previously discussed AIDE annotations, cropped JPG faces, and raw Fast
SAM3D estimations remain useful source data. The exact-cache workflow above
uses their processed products instead. The older `preprocess_aide.py` does not
reconstruct the campaign's interpolation/visibility policy; use the exported
caches to reproduce the current tests. No raw EEG recordings, EPQ responses,
driving telemetry, experiment logs, or historical trained models are needed
when the exported inputs and subset definitions are available.

## Export the input bundle on the original machine

From this repository's `MDERNet` directory:

```bash
python current/export_inputs.py \
  --source-root /data/gianluca/scripts/MDERNet \
  --subset-root /data/gianluca/scripts/outputs/preprocessing \
  --destination /data/gianluca/MDERNet-share-inputs
```

The destination must be new. The exporter copies only required input files,
checks the existing campaign hashes where available, verifies each copy, and
writes a manifest with portable relative paths. Share this folder through your
chosen file-transfer service. The recipient can put it anywhere.

## Validate and run

Set the location of the received bundle:

```bash
export MDERNET_INPUTS=/path/to/MDERNet-share-inputs
export CUDA_VISIBLE_DEVICES=0

for dataset in aide_clean aide_balanced ppb_eeg ppb_epq ppb_cluster; do
    python current/run_subsets.py --dataset "$dataset" \
        --data-root "$MDERNET_INPUTS" --check-only || break
done
```

Run the two AIDE subsets sequentially:

```bash
for dataset in aide_clean aide_balanced; do
    python -u current/run_subsets.py --dataset "$dataset" \
        --data-root "$MDERNET_INPUTS" || break
done
```

Run EEG, EPQ, and Cluster under **both** PPB-Emo conditions (six runs,
executed sequentially). The earlier five-dataset command with
`--split independent` runs only the independent condition.

```bash
for split in independent mixed; do
    for dataset in ppb_eeg ppb_epq ppb_cluster; do
        python -u current/run_subsets.py --dataset "$dataset" \
            --split "$split" --data-root "$MDERNET_INPUTS" || break 2
    done
done
```

To validate both conditions before training, add `--check-only` to the Python
command in this loop. The initial validation loop above checks PPB's default
independent condition only.

To run just one experiment, for example PPB EEG with mixed-subject splits:

```bash
python -u current/run_subsets.py --dataset ppb_eeg --split mixed \
    --data-root "$MDERNET_INPUTS"
```

Conditions have separate result directories, for example
`outputs/current/ppb_eeg_independent/` and `outputs/current/ppb_eeg_mixed/`.
AIDE results use `outputs/current/aide_clean/` and
`outputs/current/aide_balanced/`.

Results are written under `outputs/current/<dataset-and-split>/`: manifest,
status, fold splits, training histories, selected checkpoints, per-fold
validation/test metrics and predictions, and selected-fusion metadata.
Accuracy summaries can be computed as arithmetic means of the ten test-fold
scores. Completed folds are reused on restart; an interrupted fold restarts
from epoch one. The manifest rejects changed code/input bundles in an existing
run directory; use `--output-root /new/output/location` for a separate run.

The launcher never starts or stops the original campaign services. Avoid
launching it on a GPU already occupied by the current experiments.


## Complete validation-based suite (updated 7 October 2026)

The launcher now supports all 13 dataset/condition combinations:

- AIDE Full, Clean Keypoints and Balanced (stratified clip folds).
- PPB-Emo Full, Clean Keypoints, EEG, EPQ and Cluster, each with independent
  and mixed-subject folds.

The same 50-epoch protocol, training-only filtering and validation checkpoint
selection described above apply throughout. AIDE LR is 0.01; PPB LR is 0.0001.
The archived root-level ablation scripts are not used by this suite.
The original AIDE per-fold-selected fusion is not reproduced by this launcher;
it implements the fixed globally validation-selected fusion for every dataset.

From `MDERNet`, after activating the environment:

```bash
export MDERNET_INPUTS=/path/to/MDERNet-share-inputs
export CUDA_VISIBLE_DEVICES=0
bash current/run_all.sh --check-only
bash current/run_all.sh --output-root ./outputs/validation_all
```

The first command validates all 13 combinations without starting training.
The second executes them sequentially and stops on any failure. The output
root is separate from earlier runs whose source/input manifests have changed.
Completed configurations produce per-fold checkpoint/history/prediction files;
a completed experiment also produces `summary.csv` and `summary.json` under
`<output-root>/<dataset-and-split>/runs/<dataset-and-split>/`. Summaries are
unweighted means of test-fold metrics. To summarize partial results manually:

```bash
python current/summarize.py outputs/validation_all/ppb_full_independent/runs/ppb_full_independent
```

The updated input bundle includes `ppb_emo_clean_keypoints_subset.json` and its
manifest hash. Recipients of the older bundle must receive the updated bundle
(or that additional file together with the updated `data_manifest.json`).
The input bundle and pretrained weights remain external to GitHub.
