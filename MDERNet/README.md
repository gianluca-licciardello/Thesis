# MDERNet on AIDE with Fast SAM3D estimations

This directory packages the AIDE pipeline from MDERNet: a facial expression
branch (FEB), a body gesture branch (BGB), and their fusion. It consumes existing
Fast SAM3D estimations; it does not generate estimations or crop raw videos.

## Setup

Use Python 3.10 or 3.11. Run all commands below from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA 12.1, install the matching PyTorch wheels before the requirements:

```bash
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

CUDA is used automatically when available; `--device cpu` selects CPU training.
A CUDA GPU is recommended. Lower `--batch_size` if GPU memory is limited.
The face-cropping dependency facenet-pytorch is unnecessary for this pipeline
because the input faces are already cropped.

## Required external inputs

Obtain the AIDE annotations, cropped faces, and corresponding Fast SAM3D outputs
separately. Datasets, weights, and generated results are not included.

```text
data/
  AIDE/annotation/0001.json
  AIDE_cropped/0001/incarframes/0.jpg
  AIDE_cropped/0001/incarframes/1.jpg
  ...
  fastsam3d_aide/0001/incarframes.npz
```

Use matching four-digit clip IDs. Each annotation contains `emotion_label`:
`Anxiety`, `Peace`, `Weariness`, `Happiness`, or `Anger` (classes 0–4).
Each NPZ contains `keypoints_3d`, a float array of shape `(T, 70, 3)` in
MHR70 joint order. JPG basenames must be integers and sort in temporal order.
Face and keypoint sequences must correspond frame for frame; preprocessing
truncates both to the shorter length. Normally clips contain 45 frames.

The preprocessor samples 30 frames, converts faces to equalised grayscale
112×112 images, keeps 15 upper-body joints, normalises xyz per clip, and adds
visibility from finite coordinates. Its outputs are `faces/*.npy`,
`body/*.npy`, and `labels.csv`.

```bash
python preprocess_aide.py \
  --annotation_dir /path/to/AIDE/annotation \
  --cropped_dir /path/to/AIDE_cropped \
  --fastsam_dir /path/to/fastsam3d_aide \
  --out_dir ./preprocessed_aide
```

Check the reported clip counts: missing faces or estimations cause clips to be
skipped. Add `--overwrite` to regenerate an existing cache after input changes.

## Train and evaluate

A short one-fold run of the body and fusion variants:

```bash
python ablation_aide.py --subset full --mder_only \
  --preproc_dir ./preprocessed_aide --epochs 1 --fold_id 0 \
  --batch_size 2 --no_pretrained --run_dir ./outputs/smoke
```

Run all variants across ten folds:

```bash
python ablation_aide.py --subset full --all_folds --epochs 100 \
  --preproc_dir ./preprocessed_aide --batch_size 8 \
  --run_dir ./outputs/aide_full
```

By default the face backbone downloads ImageNet ResNet-18 weights. Use
`--no_pretrained` to initialise randomly without a download, or
`--pretrained_path /path/to/face_backbone.pth` for a compatible externally
provided face checkpoint. Random or ImageNet initialisation does not reproduce
experiments that used a different pretrained face checkpoint.

Results (JSON, CSV, curves, and confusion matrices) are written beneath the run
directory. Completed variants are cached: choose a new run directory when
changing settings or inputs. This runner retains its best weights in memory;
it does not export a trained checkpoint for later inference.

The included runner uses clip-level stratified cross-validation and selects
the best epoch using the held-out fold. It does not implement an independent
validation/test split and should not be treated as reproducing the final
validation campaign. AIDE has no VAD targets; dimensional metrics are not
meaningful. The default ten folds require at least ten clips per emotion.

## Paths and optional subsets

Defaults are relative to this directory, independent of the working directory.
Each path in `config.py` can be overridden before launch with its `MDERNET_`
environment variable, for example `MDERNET_OUTPUT_DIR` or
`MDERNET_AIDE_PREPROCESSED_DIR`. The preprocessing flags and training
`--preproc_dir` allow explicit paths as shown above.

`--subset full` needs no subset metadata. For `balanced` or `clean`, supply the
original subset JSON through `MDERNET_AIDE_BALANCED_SUBSET` or
`MDERNET_AIDE_CLEAN_SUBSET`. JSON format: `{"kept_clip_ids": ["0001", "0002"]}`.
These subsets filter training clips only; held-out folds remain full.

## Source files

- `preprocess_aide.py`: aligned face and keypoint preprocessing.
- `dataset_aide.py`: tensors, labels, and stratified folds.
- `model.py`: shared MDERNet architecture, including the body branch.
- `evaluate.py`: losses and metrics.
- `ablation_aide.py`: AIDE training, ablations, and result figures.
- `config.py`: architecture, training defaults, and configurable paths.

Local portability changes replace machine-specific paths and pass
`--preproc_dir` through to the training datasets. The original model and
experimental protocol are otherwise retained.
