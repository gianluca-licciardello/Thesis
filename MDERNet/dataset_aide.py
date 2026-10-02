"""
dataset_aide.py - PyTorch Dataset for the AIDE dataset (Body Gesture Branch).

Expects preprocess_aide.py to have been run first.  Loads pre-extracted face
arrays and body-keypoint arrays from the <AIDE_PREPROCESSED_DIR> directory.

Usage example
-------------
from dataset_aide import AIDEDataset, build_kfold_splits_aide

df, splits = build_kfold_splits_aide(subset="balanced")
train_ds = AIDEDataset(df, splits[0][0])
test_ds  = AIDEDataset(df, splits[0][1])
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from config import Config


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class AIDEDataset(Dataset):
    """
    One sample = (face_frames, body_data, discrete_label, dim_labels).

    face_frames   : Tensor (k, 1, H, W)      float32  normalised to [0, 1]
    body_data     : Tensor (k, J, 4)          float32  z-score normalised
    discrete_label: Tensor ()                 int64    in {0, 1, 2, 3, 4}
    dim_labels    : Tensor (3,)               float32  zeros (AIDE has no VAD)

    The 4-tuple shape matches PPBEmoDataset so the same DataLoader loops and
    evaluate_model / evaluate_model_feb functions work unmodified on AIDE.
    The dim_labels zeros are handled by setting lambda_mse=0, lambda_ccc=0
    during AIDE training and evaluation.
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        indices:     np.ndarray | list,
        preproc_dir: str = Config.AIDE_PREPROCESSED_DIR,
    ):
        self.df          = df.iloc[indices].reset_index(drop=True)
        self.preproc_dir = preproc_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row     = self.df.iloc[idx]
        clip_id = str(row.clip_id).zfill(4)

        # ── Face frames ──────────────────────────────────────────────────────
        face_path = os.path.join(self.preproc_dir, "faces", clip_id + ".npy")
        if os.path.exists(face_path):
            faces = np.load(face_path)   # (k, H, W)  float32
        else:
            faces = np.zeros(
                (Config.NUM_FRAMES, Config.FACE_SIZE, Config.FACE_SIZE), dtype=np.float32
            )
        faces = torch.from_numpy(faces).unsqueeze(1)   # (k, 1, H, W)

        # ── Body keypoints ───────────────────────────────────────────────────
        body_path = os.path.join(self.preproc_dir, "body", clip_id + ".npy")
        if os.path.exists(body_path):
            body = np.load(body_path)   # (k, J, 3)  float32
        else:
            body = np.zeros(
                (Config.NUM_BG_FRAMES, Config.NUM_BG_JOINTS, Config.NUM_BG_FEATURES), dtype=np.float32
            )
        # New files contain (x, y, z, visible). Upgrade legacy xyz files
        # at load time so existing caches fail safely and remain readable.
        if body.shape[-1] == 3:
            visible = np.isfinite(body).all(axis=-1, keepdims=True)
            body = np.where(visible, body, 0.0)
            body = np.concatenate((body, visible.astype(np.float32)), axis=-1)
        elif body.shape[-1] != Config.NUM_BG_FEATURES:
            raise ValueError(f"Unexpected body feature count {body.shape[-1]} in {body_path}")
        body = np.nan_to_num(body, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        body = torch.from_numpy(body)   # (k, J, 4)

        # ── Labels ───────────────────────────────────────────────────────────
        discrete = torch.tensor(int(row.discrete_label), dtype=torch.long)
        dims     = torch.zeros(3, dtype=torch.float32)   # AIDE has no VAD annotations

        return faces, body, discrete, dims


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation split builder
# ──────────────────────────────────────────────────────────────────────────────

def build_kfold_splits_aide(
    preproc_dir:  str  = Config.AIDE_PREPROCESSED_DIR,
    k_folds:      int  = Config.K_FOLDS,
    subset:       str  = None,
    random_state: int  = 42,
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Stratified k-fold cross-validation split for AIDE.

    AIDE has no participant structure, so splits are clip-level stratified on
    the discrete emotion label (5 classes).  StratifiedKFold with random_state=42
    guarantees all classes appear in every fold's test set.

    Parameters
    ----------
    preproc_dir  : directory produced by preprocess_aide.py
    k_folds      : number of folds (default 10)
    subset       : None        -> use all clips for both train and test
                   "balanced"  -> filter training indices to aide_balanced_subset.json;
                                  test indices always use the full held-out fold
                   "clean"     -> same, using aide_clean_keypoints_subset.json
    random_state : RNG seed for StratifiedKFold

    Returns
    -------
    df     : Full (or subset-filtered) label DataFrame.
    splits : List of (train_indices, test_indices) arrays, one per fold.
    """
    label_csv = os.path.join(preproc_dir, "labels.csv")
    if not os.path.exists(label_csv):
        raise FileNotFoundError(
            f"Labels file not found: {label_csv}\n"
            "Please run preprocess_aide.py first."
        )

    # Preserve IDs such as "0001" so labels, subset JSON files, and .npy
    # filenames use one canonical representation.
    df = pd.read_csv(label_csv, dtype={"clip_id": str})
    df["clip_id"] = df["clip_id"].str.zfill(4)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    splits = [
        (train_idx, test_idx)
        for train_idx, test_idx in skf.split(np.arange(len(df)), df["discrete_label"].values)
    ]

    if subset is not None:
        if subset == "balanced":
            subset_path = Config.AIDE_BALANCED_SUBSET
        elif subset == "clean":
            subset_path = Config.AIDE_CLEAN_SUBSET
        else:
            raise ValueError(f"Unknown subset '{subset}'. Use None, 'balanced', or 'clean'.")

        with open(subset_path) as f:
            subset_data = json.load(f)
        kept_ids = set(subset_data["kept_clip_ids"])

        # Build a boolean mask: which rows of df are in the subset
        in_subset = np.array(df["clip_id"].astype(str).isin(kept_ids))

        # Filter only the training indices; test indices are always the full fold
        splits = [
            (train_idx[in_subset[train_idx]], test_idx)
            for train_idx, test_idx in splits
        ]

    return df, splits


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for sub in [None, "balanced", "clean"]:
        tag = sub or "full"
        try:
            df, splits = build_kfold_splits_aide(subset=sub)
        except FileNotFoundError as e:
            print(f"[{tag}] {e}")
            continue

        print(f"\n[{tag}]  Total samples: {len(df)}")
        for i, (tr, te) in enumerate(splits[:3]):
            print(f"  Fold {i:2d}  train={len(tr):5d}  test={len(te):5d}")
        print("  ...")

    print("\nTesting first fold (full) …")
    try:
        df, splits = build_kfold_splits_aide(subset=None)
        ds = AIDEDataset(df, splits[0][0])
        faces, body, disc, dims = ds[0]
        print(f"  faces  : {faces.shape}  dtype={faces.dtype}")
        print(f"  body   : {body.shape}   dtype={body.dtype}")
        print(f"  label  : discrete={disc.item()}  dims={dims.tolist()}")
    except FileNotFoundError as e:
        print(e)
