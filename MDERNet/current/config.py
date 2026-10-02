import os
import torch


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # --- Paths ---------------------------------------------------------------
    DATA_ROOT          = os.environ.get("MDERNET_DATA_ROOT", os.path.join(PROJECT_DIR, "data/PPB-EMO"))
    FACE_VIDEO_SUBDIR  = "Facial_expression_data/video-face-CIR"
    DRIVING_SUBDIR     = "Driving_behavioural_data"
    LABEL_FILE         = "Psychological_data/Emotion_label.xlsx"
    PREPROCESSED_DIR   = os.environ.get("MDERNET_PREPROCESSED_DIR", os.path.join(PROJECT_DIR, "preprocessed"))
    OUTPUT_DIR         = os.environ.get("MDERNET_OUTPUT_DIR", os.path.join(PROJECT_DIR, "outputs"))

    # --- AIDE dataset --------------------------------------------------------
    AIDE_CROPPED_DIR      = os.environ.get("MDERNET_AIDE_CROPPED_DIR", os.path.join(PROJECT_DIR, "data/AIDE_cropped"))
    AIDE_ANNOTATION_DIR   = os.environ.get("MDERNET_AIDE_ANNOTATION_DIR", os.path.join(PROJECT_DIR, "data/AIDE/annotation"))
    AIDE_FASTSAM_DIR      = os.environ.get("MDERNET_AIDE_FASTSAM_DIR", os.path.join(PROJECT_DIR, "data/fastsam3d_aide"))
    AIDE_PREPROCESSED_DIR = os.environ.get("MDERNET_AIDE_PREPROCESSED_DIR", os.path.join(PROJECT_DIR, "preprocessed_aide"))
    AIDE_BALANCED_SUBSET  = os.environ.get("MDERNET_AIDE_BALANCED_SUBSET", os.path.join(PROJECT_DIR, "data/aide_balanced_subset.json"))
    AIDE_CLEAN_SUBSET     = os.environ.get("MDERNET_AIDE_CLEAN_SUBSET", os.path.join(PROJECT_DIR, "data/aide_clean_keypoints_subset.json"))

    # Emotion encoding follows UbH-GCN/gen_data_fastsam3d.py
    AIDE_EMOTION_MAP = {
        "Anxiety": 0, "Peace": 1, "Weariness": 2, "Happiness": 3, "Anger": 4,
        "happiness": 3, "weariness": 2, "forward": 1,  # lowercase/alias variants
    }
    AIDE_EMOTION_NAMES = ["Anxiety", "Peace", "Weariness", "Happiness", "Anger"]
    AIDE_NUM_CLASSES   = 5

    # --- Body Gesture Branch (BGB) -------------------------------------------
    # Three pruning steps applied to MHR-70's 70 joints:
    #
    # 1. Face pruning: joints 0-4 (nose, left_eye, right_eye, left_ear, right_ear)
    #    are face landmarks that carry no body-gesture signal.
    #
    # 2. All-finger pruning: remove all finger joints.
    #    Right hand fingers: joints 21-40 (5 fingers × 4 joints = 20 joints).
    #    Left hand fingers:  joints 42-61 (5 fingers × 4 joints = 20 joints).
    #    Wrists (joint 41 = right_wrist, joint 62 = left_wrist) are KEPT.
    #
    # 3. Lower-body pruning: drop joints below the waist.
    #    Hips (9, 10) are the waist boundary and are kept.
    #    Joints 11-20: knees, ankles, toes, heels → removed.
    #
    # Joints 63-69 (olecranons, cubital fossae, acromions, neck) are kept.
    #
    # 70 − 5 (face) − 20 (right fingers) − 20 (left fingers) − 10 (lower body)
    #   = 15 joints kept: {5-10, 41, 62, 63-69}
    _BGB_FACE_REMOVE       = set(range(0, 5))        # nose, eyes, ears
    _BGB_FINGER_REMOVE     = set(range(21, 41)) | set(range(42, 62))   # all finger joints
    _BGB_LOWER_BODY_REMOVE = set(range(11, 21))      # knees, ankles, feet
    _BGB_REMOVE            = _BGB_FACE_REMOVE | _BGB_FINGER_REMOVE | _BGB_LOWER_BODY_REMOVE
    MHR70_KEEP_INDICES     = sorted(set(range(70)) - _BGB_REMOVE)   # 15 elements

    NUM_BG_FRAMES  = 30    # must equal NUM_FRAMES for direct FAM alignment with face frames
    NUM_BG_JOINTS  = 15    # after face + all-finger + lower-body pruning
    NUM_BG_FEATURES = 4    # stored channels: normalised x, y, z + binary joint visibility
    # Positions inside MHR70_KEEP_INDICES: five edges from bones.csv, plus
    # the requested neck-to-left/right-shoulder upper-torso relations.
    BGB_BONE_PAIRS = (
        (4, 5),                    # left hip -> right hip
        (0, 2), (1, 3),           # shoulders -> elbows
        (2, 7), (3, 6),           # elbows -> wrists
        (14, 0), (14, 1),         # neck -> left/right shoulder
    )
    NUM_BG_BONES = len(BGB_BONE_PAIRS)
    BGB_SCHEMA = "mhr70-15_xyz_visibility_7bones_v2"
    BG_HIDDEN_DIM  = 512   # mirrors DB_HIDDEN_DIM
    BG_FEAT_DIM    = 256   # mirrors DB_FEAT_DIM

    # MS-Celeb-1M pre-training paths (used by pretrain_msceleb.py)
    MSCELEB_DIR        = os.environ.get("MDERNET_MSCELEB_DIR", os.path.join(PROJECT_DIR, "data/MS-Celeb-1M"))
    PRETRAINED_FEFEM   = os.environ.get("MDERNET_PRETRAINED_FEFEM", os.path.join(PROJECT_DIR, "outputs/pretrained_fefem.pth"))
    PRETRAIN_EPOCHS    = 30
    PRETRAIN_LR        = 0.01
    PRETRAIN_BATCH     = 256

    # --- Preprocessing -------------------------------------------------------
    FACE_SIZE            = 112    # pixels (square)
    NUM_FRAMES           = 30     # frames uniformly sampled per clip
    VIDEO_SKIP_DURATION  = 0.0    # seconds to skip at start (0 = use first 15 s, per paper)
    VIDEO_DURATION       = 15.0   # seconds of content to use after the skip

    # --- PPB-Emo body keypoints (Fast SAM3D, used by BGB branch) ---------------
    PPB_FASTSAM_DIR = os.environ.get("MDERNET_PPB_FASTSAM_DIR", os.path.join(PROJECT_DIR, "data/fastsam3d_ppb_emo"))
    PPB_BODY_FPS    = 30   # fastsam3d clips are at 30 fps (900 frames = 30 s)
    PPB_BODY_WINDOW = 450  # first 15 s × 30 fps = 450 frames out of 900

    # --- Driving behaviour ---------------------------------------------------
    # "Steering wheel position" is converted to rotational speed via np.diff.
    DB_RAW_COLS = [
        "Steering wheel position",
        "Gas pedal position",
        "Brake pedal force",
        "Velocity",
        "Acceleration",
        "Lateral velocity",
        "Lateral acceleration",
    ]
    NUM_DB_FEATURES    = 7
    NUM_DB_SAMPLES     = 300

    # --- Model architecture --------------------------------------------------
    FEAT_DIM           = 512      # FEFEM (ResNet-18) output dimension
    FAM_REDUCTION      = 32       # FAM hidden = FEAT_DIM // FAM_REDUCTION = 16
    DB_HIDDEN_DIM      = 512
    DB_FEAT_DIM        = 256
    NUM_CLASSES        = 7        # discrete emotions
    DIM_OUTPUT         = 3        # valence, arousal, dominance

    # --- Training ------------------------------------------------------------
    BATCH_SIZE         = 64    # paper value; TITAN Xp 12 GB handles 64x30 ResNet-18 frames
    LR                 = 0.01   # paper does not report the initial learning rate
    MOMENTUM           = 0.9
    NESTEROV           = True
    WEIGHT_DECAY       = 1e-4
    NUM_EPOCHS         = 50
    K_FOLDS            = 10
    RANDOM_SEED        = 42
    EVAL_SCHEMA        = "cv10_train8_val1_test1_valckpt_seed42_lr001_v2"
    T_0                = 5        # CosineAnnealingWarmRestarts period (10 cycles over 50 epochs)

    LAMBDA_MSE         = 0.1   # reduced: MSE magnitude (5–10) was dominating CE (~2)
    LAMBDA_CCC         = 1.0   # increased: encourages correlated dimensional predictions
    LAMBDA_F1          = 1.0   # soft macro F1 loss weight (AIDE only; 0 for PPB-EMO)

    # --- Driving behaviour filtering -----------------------------------------
    BINARY_THRESHOLD   = 0.3   # kept for reference; superseded by soft masking (see model.py)

    # --- Label mappings ------------------------------------------------------
    EMOTION_MAP = {
        "AD":  0,
        "DD":  1,
        "FD":  2,
        "HD":  3,
        "ND":  4,
        "SAD": 5,
        "SD":  6,
    }
    EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

    # --- Device --------------------------------------------------------------
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # --- Runtime directory creation ------------------------------------------
    @classmethod
    def ensure_dirs(cls):
        for d in (cls.PREPROCESSED_DIR, cls.OUTPUT_DIR):
            os.makedirs(d, exist_ok=True)

    @classmethod
    def next_run_dir(cls, base_dir: str) -> str:
        """Create and return the next outputs/run_N directory."""
        i = 1
        while os.path.exists(os.path.join(base_dir, f"run_{i}")):
            i += 1
        path = os.path.join(base_dir, f"run_{i}")
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def latest_run_dir(cls, base_dir: str):
        """Return the highest-numbered outputs/run_N that exists, or None."""
        i, last = 1, None
        while os.path.exists(os.path.join(base_dir, f"run_{i}")):
            last = os.path.join(base_dir, f"run_{i}")
            i += 1
        return last
