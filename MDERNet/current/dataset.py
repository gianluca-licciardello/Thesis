"""
dataset.py – PyTorch Dataset for the PPB-EMO dataset.

Expects the preprocessing step (preprocess.py) to have been run first.
Loads pre-extracted face-frame arrays and driving-behaviour arrays from the
<PREPROCESSED_DIR> directory.

Usage example
-------------
from dataset import PPBEmoDataset, build_kfold_splits

df_labels, splits = build_kfold_splits()   # returns (full DataFrame, list of (train_idx, test_idx))
train_ds = PPBEmoDataset(df_labels, splits[0][0])
test_ds  = PPBEmoDataset(df_labels, splits[0][1])
"""

import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset

from config import Config


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class PPBEmoDataset(Dataset):
    """
    One sample = (face_frames, driving_data, discrete_label, dim_labels).

    face_frames   : Tensor (k, 1, H, W)   float32  normalised to [0, 1]
    driving_data  : Tensor (N_db, 7)      float32  z-score normalised
    discrete_label: Tensor ()             int64
    dim_labels    : Tensor (3,)           float32  [valence, arousal, dominance]
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        indices:     np.ndarray | list,
        preproc_dir: str = Config.PREPROCESSED_DIR,
    ):
        """
        Parameters
        ----------
        df       : Full label DataFrame (returned by load_label_table or read from labels.csv).
        indices  : Row indices (into df) to include in this split.
        preproc_dir : Directory containing faces/ and driving/ sub-folders.
        """
        self.df          = df.iloc[indices].reset_index(drop=True)
        self.preproc_dir = preproc_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = f"{row.participant}_{row.emotion_code}"

        # ── Face frames ──────────────────────────────────────────────────
        face_path = os.path.join(self.preproc_dir, "faces", key + ".npy")
        if os.path.exists(face_path):
            faces = np.load(face_path)   # (k, H, W)  float32
        else:
            faces = np.zeros(
                (Config.NUM_FRAMES, Config.FACE_SIZE, Config.FACE_SIZE),
                dtype=np.float32,
            )

        # Add channel dimension: (k, H, W) → (k, 1, H, W)
        faces = torch.from_numpy(faces).unsqueeze(1)   # (k, 1, H, W)

        # ── Driving behaviour ─────────────────────────────────────────────
        db_path = os.path.join(self.preproc_dir, "driving", key + ".npy")
        if os.path.exists(db_path):
            db = np.load(db_path)    # (N_db, 7)  float32
        else:
            db = np.zeros((Config.NUM_DB_SAMPLES, Config.NUM_DB_FEATURES), dtype=np.float32)
        db = torch.from_numpy(db)    # (N_db, 7)

        # ── Labels ────────────────────────────────────────────────────────
        discrete = torch.tensor(row.discrete_label, dtype=torch.long)
        dims = torch.tensor(
            [row.valence, row.arousal, row.dominance], dtype=torch.float32
        )

        return faces, db, discrete, dims


# ──────────────────────────────────────────────────────────────────────────────
# BGB dataset: PPB-Emo face + body keypoints (replaces driving signal)
# ──────────────────────────────────────────────────────────────────────────────

class PPBEmoBGBDataset(Dataset):
    """
    PPB-Emo dataset variant for the Body Gesture Branch (BGB).

    Returns the same 4-tuple as PPBEmoDataset but with body keypoints
    instead of driving data as the secondary signal:

    face_frames   : Tensor (k, 1, H, W)         float32  normalised to [0, 1]
    body_keypoints: Tensor (k, J, 4)             float32  z-score normalised  (k=30, J=15)
    discrete_label: Tensor ()                    int64
    dim_labels    : Tensor (3,)                  float32  [valence, arousal, dominance]

    Prerequisites
    -------------
    Run preprocess_ppb_body.py first to populate preproc_dir/body/{key}.npy.
    If a body file is missing the sample falls back to an all-zero tensor so
    training does not crash, but that sample will contribute no body signal.
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        indices:     np.ndarray | list,
        preproc_dir: str = Config.PREPROCESSED_DIR,
    ):
        self.df          = df.iloc[indices].reset_index(drop=True)
        self.preproc_dir = preproc_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        key = f"{row.participant}_{row.emotion_code}"

        # ── Face frames (same as PPBEmoDataset) ──────────────────────────────
        face_path = os.path.join(self.preproc_dir, "faces", key + ".npy")
        if os.path.exists(face_path):
            faces = np.load(face_path)   # (k, H, W) float32
        else:
            faces = np.zeros(
                (Config.NUM_FRAMES, Config.FACE_SIZE, Config.FACE_SIZE),
                dtype=np.float32,
            )
        faces = torch.from_numpy(faces).unsqueeze(1)   # (k, 1, H, W)

        # ── Body keypoints ────────────────────────────────────────────────────
        body_path = os.path.join(self.preproc_dir, "body", key + ".npy")
        if os.path.exists(body_path):
            body = np.load(body_path)    # (k, J, 3) float32
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

        # ── Labels ────────────────────────────────────────────────────────────
        discrete = torch.tensor(row.discrete_label, dtype=torch.long)
        dims = torch.tensor(
            [row.valence, row.arousal, row.dominance], dtype=torch.float32
        )

        return faces, body, discrete, dims


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation split builder
# ──────────────────────────────────────────────────────────────────────────────

def build_kfold_splits(
    preproc_dir=Config.PREPROCESSED_DIR, k_folds=Config.K_FOLDS,
    mix_subjects=False, downsample_train=False, subsets_path=None,
    subset_name="Cluster", clean_subset_path=None, with_validation=False,
    random_state=Config.RANDOM_SEED,
):
    """Build paper-style participant folds or stratified clip folds.

    Legacy callers receive (train, test).  ``with_validation=True`` returns
    deterministic (train, validation, test) folds using an 8/1/1 rotation:
    test=i, validation=(i+1) mod k, training=the remaining eight folds.
    Reliability/clean subsets filter training indices only.
    """
    label_csv = os.path.join(preproc_dir, "labels.csv")
    if not os.path.exists(label_csv):
        raise FileNotFoundError(f"Labels file not found: {label_csv}\nPlease run preprocess.py first.")
    df = pd.read_csv(label_csv)
    if mix_subjects:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        test_folds = [te for _, te in skf.split(np.arange(len(df)), df["discrete_label"].values)]
    else:
        participants = sorted(df["participant"].unique())
        if len(participants) < k_folds:
            raise ValueError(f"Only {len(participants)} participants but k_folds={k_folds}")
        test_folds = [np.where(df["participant"].isin(participants[i::k_folds]).values)[0]
                      for i in range(k_folds)]
    if with_validation:
        splits=[]
        for i in range(k_folds):
            vi=(i+1)%k_folds
            tr=np.sort(np.concatenate([test_folds[j] for j in range(k_folds) if j not in (i,vi)]))
            splits.append((tr, np.sort(test_folds[vi]), np.sort(test_folds[i])))
    else:
        all_idx=np.arange(len(df))
        splits=[(np.setdiff1d(all_idx,te,assume_unique=True),te) for te in test_folds]
    train_mask=np.ones(len(df),dtype=bool)
    if downsample_train:
        if not subsets_path or not os.path.exists(subsets_path):
            raise FileNotFoundError(f"subsets_path not found: {subsets_path}")
        with open(subsets_path) as f: subsets_data=json.load(f)
        if subset_name not in subsets_data.get("subsets",{}):
            raise ValueError(f"Subset '{subset_name}' not found; available: {list(subsets_data.get('subsets',{}))}")
        keys={f"{x['participant']}-{x['category']}" for x in subsets_data["subsets"][subset_name]}
        train_mask=np.array([f"{r.participant}-{r.emotion_code}" in keys for _,r in df.iterrows()])
    if clean_subset_path is not None:
        with open(clean_subset_path) as f: clean=json.load(f)
        keys=set(clean["kept_clip_ids"])
        train_mask &= np.array([f"{r.participant}-{r.emotion_code}" in keys for _,r in df.iterrows()])
    splits=[(sp[0][train_mask[sp[0]]],*sp[1:]) for sp in splits]
    return df, splits


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, splits = build_kfold_splits()
    print(f"Total samples : {len(df)}")
    print(f"Folds         : {len(splits)}")
    for i, (tr, te) in enumerate(splits):
        print(f"  Fold {i:2d}  train={len(tr):4d}  test={len(te):4d}")

    print("\nTesting first fold …")
    train_ds = PPBEmoDataset(df, splits[0][0])
    item     = train_ds[0]
    faces, db, disc, dims = item
    print(f"  faces  : {faces.shape}  dtype={faces.dtype}")
    print(f"  db     : {db.shape}    dtype={db.dtype}")
    print(f"  label  : discrete={disc.item()}  dims={dims.tolist()}")
