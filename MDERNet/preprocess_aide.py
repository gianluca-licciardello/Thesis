"""
preprocess_aide.py - Offline preprocessing for the AIDE dataset.

Run this script ONCE before training with the Body Gesture Branch (BGB).
For each AIDE clip it:
  1. Loads the 45 already-cropped face JPGs from AIDE_cropped/{clip_id}/incarframes/
  2. Converts each to grayscale 112×112, applies histogram equalisation, normalises to [0,1]
  3. Loads Fast SAM3D body keypoints: fastsam3d_aide/{clip_id}/incarframes.npz -> keypoints_3d
  4. Removes face, all finger, and lower-body joints → 15 upper-body joints remain
  5. Z-score normalises the keypoints per clip (same strategy as driving data in preprocess.py)
  6. Uniformly samples 30 frames from the 45 available (same frame indices for face + body)
  7. Saves face arrays as (30, 112, 112) and body arrays as (30, 15, 4): x/y/z + visibility
  8. Writes a labels.csv

Output layout
-------------
preprocessed_aide/
  faces/
    0001.npy     # (30, 112, 112) float32  [0, 1]
    ...
  body/
    0001.npy     # (30, 15, 4)   float32  xyz + visibility
    ...
  labels.csv     # columns: clip_id, discrete_label

Usage
-----
  python preprocess_aide.py
  python preprocess_aide.py --overwrite          # reprocess all clips
  python preprocess_aide.py --out_dir <path>     # custom output directory
"""

import argparse
import json
import logging
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Indices for the 15 retained non-hand upper-body MHR70 joints
KEEP_IDX = np.array(Config.MHR70_KEEP_INDICES, dtype=np.int64)   # (15,)

# Number of output frames (must match Config.NUM_FRAMES / Config.NUM_BG_FRAMES)
NUM_OUT_FRAMES = Config.NUM_BG_FRAMES   # 30


def load_aide_labels(annotation_dir: str) -> pd.DataFrame:
    """
    Parse every AIDE annotation JSON and return a DataFrame of valid clips.

    Skips clips with parse errors, missing emotion_label fields,
    or unrecognised emotion strings.
    """
    records = []
    skipped = 0
    for fname in sorted(os.listdir(annotation_dir)):
        if not fname.endswith(".json"):
            continue
        clip_id = fname[:-5]   # e.g. "0001"
        path = os.path.join(annotation_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            skipped += 1
            continue

        raw_label = data.get("emotion_label", "")
        label_int = Config.AIDE_EMOTION_MAP.get(raw_label)
        if label_int is None:
            skipped += 1
            continue

        records.append({"clip_id": clip_id, "discrete_label": label_int})

    df = pd.DataFrame(records)
    log.info("Labels: %d valid clips, %d skipped", len(df), skipped)
    return df


def load_face_frames(clip_id: str, cropped_dir: str = Config.AIDE_CROPPED_DIR,
                     face_size: int = Config.FACE_SIZE) -> np.ndarray | None:
    """
    Load all face JPGs for a clip, sort by frame index, return (T, face_size, face_size) float32.

    Returns None if the directory is missing or empty.
    """
    frame_dir = os.path.join(cropped_dir, clip_id, "incarframes")
    if not os.path.isdir(frame_dir):
        return None

    files = [f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")]
    if not files:
        return None

    # Sort numerically (filenames are integers without zero-padding)
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    frames = []
    for fname in files:
        bgr = cv2.imread(os.path.join(frame_dir, fname))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (face_size, face_size), interpolation=cv2.INTER_AREA)
        eq = cv2.equalizeHist(resized).astype(np.float32) / 255.0
        frames.append(eq)

    if not frames:
        return None
    return np.stack(frames, axis=0)   # (T, H, W)


def load_body_keypoints(clip_id: str,
                        fastsam_dir: str = Config.AIDE_FASTSAM_DIR) -> np.ndarray | None:
    """
    Load Fast SAM3D keypoints for a clip and retain 15 non-hand upper-body joints.

    Returns (T, 15, 3) float32 or None if the file is missing.
    """
    npz_path = os.path.join(fastsam_dir, clip_id, "incarframes.npz")
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)
    kps = data["keypoints_3d"].astype(np.float32)   # (T, 70, 3)
    if kps.ndim != 3 or kps.shape[1] != 70:
        return None

    return kps[:, KEEP_IDX, :]   # (T, 15, 3)


def normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    """
    Per-clip z-score normalisation: subtract global mean, divide by global std.

    kps: (T, J, 3) float32 -> (T, J, 4) float32. The last channel is
    one only when all three source coordinates are finite.
    Mirrors the per-clip normalisation applied to driving data in preprocess.py.
    """
    visible = np.isfinite(kps).all(axis=-1, keepdims=True)
    valid_xyz = np.where(visible, kps, np.nan)
    mu = np.nanmean(valid_xyz, axis=(0, 1), keepdims=True)
    sigma = np.nanstd(valid_xyz, axis=(0, 1), keepdims=True)
    mu = np.nan_to_num(mu, nan=0.0)
    sigma = np.nan_to_num(sigma, nan=1.0)
    normalized = np.where(visible, (kps - mu) / (sigma + 1e-8), 0.0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((normalized, visible.astype(np.float32)), axis=-1)


def sample_frames(arr: np.ndarray, n: int = NUM_OUT_FRAMES) -> np.ndarray:
    """
    Uniformly sample n frames from arr along axis 0.
    arr: (T, ...) -> (n, ...)
    """
    T = arr.shape[0]
    indices = np.linspace(0, T - 1, n, dtype=int)
    return arr[indices]


def run_preprocessing(
    annotation_dir: str = Config.AIDE_ANNOTATION_DIR,
    cropped_dir:    str = Config.AIDE_CROPPED_DIR,
    fastsam_dir:    str = Config.AIDE_FASTSAM_DIR,
    out_dir:        str = Config.AIDE_PREPROCESSED_DIR,
    overwrite:      bool = False,
):
    os.makedirs(os.path.join(out_dir, "faces"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "body"),  exist_ok=True)

    df = load_aide_labels(annotation_dir)

    ok = fail_face = fail_body = skipped = 0
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing AIDE"):
        clip_id = row.clip_id
        face_out = os.path.join(out_dir, "faces", clip_id + ".npy")
        body_out = os.path.join(out_dir, "body",  clip_id + ".npy")

        if not overwrite and os.path.exists(face_out) and os.path.exists(body_out):
            records.append({"clip_id": clip_id, "discrete_label": row.discrete_label})
            skipped += 1
            continue

        # Face frames
        faces_raw = load_face_frames(clip_id, cropped_dir)
        if faces_raw is None:
            fail_face += 1
            log.debug("Missing face frames: %s", clip_id)
            continue

        # Body keypoints
        body_raw = load_body_keypoints(clip_id, fastsam_dir)
        if body_raw is None:
            fail_body += 1
            log.debug("Missing body keypoints: %s", clip_id)
            continue

        # Sample 30 frames (same indices for both face and body for temporal alignment)
        T = min(faces_raw.shape[0], body_raw.shape[0])
        face_sampled = sample_frames(faces_raw[:T], NUM_OUT_FRAMES)   # (30, H, W)
        body_sampled = sample_frames(body_raw[:T],  NUM_OUT_FRAMES)   # (30, 15, 3)
        body_sampled = normalize_keypoints(body_sampled)

        np.save(face_out, face_sampled)
        np.save(body_out, body_sampled)
        records.append({"clip_id": clip_id, "discrete_label": row.discrete_label})
        ok += 1

    labels_df = pd.DataFrame(records)
    labels_df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)

    log.info(
        "Done. OK=%d  skipped=%d  face_fail=%d  body_fail=%d  total_in_csv=%d",
        ok, skipped, fail_face, fail_body, len(labels_df),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIDE offline preprocessing (BGB pipeline)")
    parser.add_argument("--annotation_dir", default=Config.AIDE_ANNOTATION_DIR)
    parser.add_argument("--cropped_dir",    default=Config.AIDE_CROPPED_DIR)
    parser.add_argument("--fastsam_dir",    default=Config.AIDE_FASTSAM_DIR)
    parser.add_argument("--out_dir",        default=Config.AIDE_PREPROCESSED_DIR)
    parser.add_argument("--overwrite",      action="store_true")
    args = parser.parse_args()

    run_preprocessing(
        annotation_dir = args.annotation_dir,
        cropped_dir    = args.cropped_dir,
        fastsam_dir    = args.fastsam_dir,
        out_dir        = args.out_dir,
        overwrite      = args.overwrite,
    )
