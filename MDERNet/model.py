"""
MDERNet - Multimodal Driver Emotion Recognition Network

Dual-branch architecture (facial expression + secondary branch) with
middle-level feature fusion, as described in:
  Wu et al., "AI-enabled intelligent cockpit proactive affective interaction:
  middle-level feature fusion dual-branch deep learning network for driver
  emotion recognition", Adv. Manuf. 13, 525-538 (2025).

Module roles
------------
FEFEM  - ResNet-18 backbone (1-ch grayscale input), outputs 512-d per frame
FM     - Softmax temporal attention; fuses k frames -> 1 vector
         Linear(D->1) score, softmax over frames, weighted sum + BN + ReLU
FAM    - SE-style frame attention; scores each frame vs the video-level feature
         FC1(1024->16) + ReLU + FC2(16->512) + Sigmoid, reduction r=32
DBFEM  - MLP(2100->512->512->256) on flattened refined driving data   [branch="driving"]
BGFEM  - MLP(1350->512->512->256) on flattened refined body keypoints [branch="body"]
DM     - Two parallel FC heads (discrete N-class + dimensional 3-dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import Config


# ---------------------------------------------------------------------------
# FEFEM  (ResNet-18 without classification head)
# ---------------------------------------------------------------------------

class FEFEM(nn.Module):
    """Extracts a 512-d feature vector from a 112x112 grayscale face image."""

    _RESNET_TO_BACKBONE = {
        "conv1": "0",
        "bn1": "1",
        "layer1": "4",
        "layer2": "5",
        "layer3": "6",
        "layer4": "7",
    }

    @classmethod
    def _canonicalise_pretrained_state(cls, checkpoint):
        """Convert supported ResNet/FEFEM checkpoints to ``self.backbone`` keys.

        ``pretrain_msceleb.py`` saves an FEFEM state dict with ``backbone.*``
        keys, whereas the external Dominik checkpoint is a torchvision-style
        ResNet state dict (``conv1.*``, ``bn1.*``, ``layer1.*``, ...). The
        runtime backbone is an ``nn.Sequential`` and therefore expects numeric
        keys (``0.*``, ``1.*``, ``4.*``, ...).
        """
        if not isinstance(checkpoint, dict):
            raise TypeError("Pretrained checkpoint must contain a state dictionary.")

        state = checkpoint
        for container_key in ("state_dict", "model_state_dict"):
            candidate = state.get(container_key)
            if isinstance(candidate, dict):
                state = candidate
                break

        canonical = {}
        for original_key, value in state.items():
            key = original_key
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("fefem."):
                key = key[len("fefem."):]
            if key.startswith("backbone."):
                key = key[len("backbone."):]

            # Ignore face-recognition/full-ResNet classification heads.
            if key.startswith(("fc.", "head.", "classifier.")):
                continue

            root, dot, suffix = key.partition(".")
            if root in cls._RESNET_TO_BACKBONE:
                key = cls._RESNET_TO_BACKBONE[root] + (dot + suffix if dot else "")

            # Numeric keys are already in self.backbone's canonical format.
            if key.partition(".")[0].isdigit():
                canonical[key] = value

        return canonical

    def __init__(self, pretrained: bool = True, pretrained_path: str = None):
        """
        pretrained      - use ImageNet weights for the ResNet-18 backbone
        pretrained_path - path to MS-Celeb-1M pretrained FEFEM state_dict;
                          when set, pretrained is ignored and these weights
                          are loaded instead (face-specific initialisation)
        """
        super().__init__()
        # A supplied face-pretrained checkpoint replaces ImageNet
        # initialisation, as in the paper's MS-Celeb-1M-pretrained FEFEM.
        use_imagenet = pretrained and pretrained_path is None
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet else None
        )

        # Adapt conv1 for 1-channel input by averaging the 3 pretrained filters.
        # This preserves the learned spatial patterns while collapsing RGB -> grey.
        orig_w  = resnet.conv1.weight.data           # (64, 3, 7, 7)
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = orig_w.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv

        # All layers up to (and including) avgpool; discard the FC head.
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,   # -> (B, 512, 1, 1)
        )

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state = self._canonicalise_pretrained_state(ckpt)
            conv1_key = "0.weight"
            if conv1_key in state and state[conv1_key].ndim == 4:
                if state[conv1_key].shape[1] == 3:
                    state[conv1_key] = state[conv1_key].mean(dim=1, keepdim=True)
                elif state[conv1_key].shape[1] != 1:
                    raise ValueError(
                        f"Unsupported conv1 input channels in {pretrained_path}: "
                        f"{state[conv1_key].shape[1]}"
                    )

            incompatible = self.backbone.load_state_dict(state, strict=False)
            # Old PyTorch checkpoints may omit this derived BatchNorm counter;
            # every learned parameter and all other buffers must be present.
            missing = [
                key for key in incompatible.missing_keys
                if not key.endswith("num_batches_tracked")
            ]
            if missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Incompatible FEFEM checkpoint {pretrained_path}: "
                    f"missing={missing}, unexpected={incompatible.unexpected_keys}"
                )

            target_state = self.backbone.state_dict()
            loaded_numel = sum(
                target_state[key].numel() for key in state if key in target_state
            )
            total_numel = sum(t.numel() for t in target_state.values())
            self.pretrained_load_info = {
                "path": str(pretrained_path),
                "loaded_tensors": len(state),
                "loaded_numel": loaded_numel,
                "total_numel": total_numel,
                "missing_legacy_buffers": [
                    key for key in incompatible.missing_keys
                    if key.endswith("num_batches_tracked")
                ],
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W)  ->  (B, 512)"""
        return self.backbone(x).flatten(1)


# ---------------------------------------------------------------------------
# FM  (Fusion Module)
# ---------------------------------------------------------------------------

class FM(nn.Module):
    """
    Fuses k per-frame feature vectors into a single video-level vector.

    Lightweight temporal self-attention: a single linear scorer maps each
    frame's D-dim feature to a scalar, then softmax-normalised weights are
    used for a weighted sum of frame features.

    This replaces the original ShuffleNet-inspired depthwise temporal conv
    (k*D ≈ 15k params per FM) with a much lighter design (D ≈ 512 params),
    reducing overfitting on the small PPB-EMO dataset under participant-level CV.
    The paper specifies temporal aggregation of frame features but does not
    constrain the exact FM architecture.
    """

    def __init__(self, feat_dim: int = 512, num_frames: int = 30):
        super().__init__()
        self.score = nn.Linear(feat_dim, 1, bias=False)
        self.bn    = nn.BatchNorm1d(feat_dim)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """frame_feats: (B, k, D)  ->  (B, D)"""
        scores = torch.softmax(self.score(frame_feats).squeeze(-1), dim=1)  # (B, k)
        x = (scores.unsqueeze(-1) * frame_feats).sum(dim=1)                 # (B, D)
        return self.relu(self.bn(x))


# ---------------------------------------------------------------------------
# FAM  (Frame Attention Module)
# ---------------------------------------------------------------------------

class FAM(nn.Module):
    """
    SE-style attention that scores each frame relative to the whole video.

    For each frame i (SE excitation, reduction r=32):
        W_i  = Sigmoid( FC2( ReLU( FC1( cat(frame_i, video) ) ) ) )
              FC1: 2C -> C/r = 1024 -> 16
              FC2: C/r -> C  = 16   -> 512
        M_i  = W_i * frame_i      (element-wise channel scaling)
        w_i  = mean(W_i)          (scalar importance for driving filtering)

    Reduction increased from 16 to 32 to halve FAM parameter count and
    reduce overfitting. The paper specifies SE-style excitation but does not
    prescribe the reduction ratio.
    """

    def __init__(self, feat_dim: int = 512, reduction: int = 32):
        super().__init__()
        hidden = feat_dim // reduction   # 512 // 32 = 16
        self.fc1 = nn.Linear(feat_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, feat_dim)

    def forward(self, frame_feats: torch.Tensor, video_feat: torch.Tensor):
        """
        Returns
        -------
        W_fam    : (B, k, D)  per-channel weights in (0, 1)
        M_fam    : (B, k, D)  W_fam * frame_feats
        scalar_w : (B, k)     mean over D, used to filter driving data
        """
        B, k, D = frame_feats.shape
        v = video_feat.unsqueeze(1).expand(-1, k, -1)   # (B, k, D)
        x = torch.cat([frame_feats, v], dim=-1).view(B * k, 2 * D)

        W = torch.sigmoid(self.fc2(F.relu(self.fc1(x)))).view(B, k, D)
        return W, W * frame_feats, W.mean(dim=-1)


# ---------------------------------------------------------------------------
# DBFEM  (Driving Behaviour Feature Extraction Module)
# ---------------------------------------------------------------------------

class DBFEM(nn.Module):
    """MLP that encodes the refined (attention-filtered) driving-behaviour data."""

    def __init__(
        self,
        input_dim:  int   = Config.NUM_DB_SAMPLES * Config.NUM_DB_FEATURES,
        hidden_dim: int   = Config.DB_HIDDEN_DIM,
        feat_dim:   int   = Config.DB_FEAT_DIM,
        dropout:    float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,  hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim),   nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim)  ->  (B, feat_dim)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# BGFEM  (Body Gesture Feature Extraction Module)
# ---------------------------------------------------------------------------

class BGFEM(nn.Module):
    """Paper-style MLP encoder for flattened, structured body features.

    Per frame the full representation concatenates 15 joint xyz coordinates,
    15 joint-visibility bits, and seven 3-D bone vectors. Feature ablations alter
    only this input composition; the three-layer MLP remains unchanged.
    """

    def __init__(
        self,
        num_frames: int = Config.NUM_BG_FRAMES,
        num_joints: int = Config.NUM_BG_JOINTS,
        num_bones: int = Config.NUM_BG_BONES,
        use_visibility: bool = True,
        use_bones: bool = True,
        hidden_dim: int = Config.BG_HIDDEN_DIM,
        feat_dim: int = Config.BG_FEAT_DIM,
        dropout: float = 0.5,
    ):
        super().__init__()
        features_per_frame = num_joints * 3
        if use_visibility:
            features_per_frame += num_joints
        if use_bones:
            features_per_frame += num_bones * 3
        self.features_per_frame = features_per_frame
        input_dim = num_frames * features_per_frame
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DM  (Decision Module)
# ---------------------------------------------------------------------------

class DM(nn.Module):
    """
    Single multimodal FC layer: concat(face_feat, db_feat) -> emotion predictions.
    Two parallel heads share the fused input (no hidden layers between fusion and output).
    """

    def __init__(
        self,
        face_feat_dim: int = Config.FEAT_DIM,
        db_feat_dim:   int = Config.DB_FEAT_DIM,
        num_classes:   int = Config.NUM_CLASSES,
        dim_output:    int = Config.DIM_OUTPUT,
    ):
        super().__init__()
        total = face_feat_dim + db_feat_dim
        self.discrete_head    = nn.Linear(total, num_classes)
        self.dimensional_head = nn.Linear(total, dim_output)

    def forward(self, face_feat: torch.Tensor, db_feat: torch.Tensor):
        x = torch.cat([face_feat, db_feat], dim=-1)
        dim_out = torch.sigmoid(self.dimensional_head(x)) * 8 + 1
        return self.discrete_head(x), dim_out


# ---------------------------------------------------------------------------
# MDERNet  (full model)
# ---------------------------------------------------------------------------

class MDERNet(nn.Module):
    """
    Dual-branch model: facial expression branch + secondary branch.

    branch='driving' (default, PPB-EMO):
      1.  FEFEM     : k face images -> frame_feats (B, k, 512)
      2.  FM-1      : frame_feats -> video_feat (B, 512)
      3.  FAM       : (frame_feats, video_feat) -> M_fam (B, k, 512), scalar_w (B, k)
      4.  FM-2      : M_fam -> face_feat (B, 512)
      5.  Refinement: scalar_w interpolated k->300, soft continuous mask on driving data
      6.  DBFEM     : refined driving (B, 2100) -> db_feat (B, 256)
      7.  DM        : (face_feat, db_feat) -> discrete logits + dimensional predictions

    branch='body' (BGB, AIDE):
      Steps 1-4 identical.  Step 5 applies scalar_w directly per frame (no interpolation
      needed since body and face share the same k-frame temporal axis).
      6.  BGFEM     : xyz + optional visibility/bones -> bg_feat (B, 256)
      7.  DM        : (face_feat, bg_feat) -> discrete logits + dimensional predictions
    """

    def __init__(
        self,
        num_frames:      int   = Config.NUM_FRAMES,
        feat_dim:        int   = Config.FEAT_DIM,
        fam_reduction:   int   = Config.FAM_REDUCTION,
        num_classes:     int   = Config.NUM_CLASSES,
        dim_output:      int   = Config.DIM_OUTPUT,
        pretrained:      bool  = True,
        pretrained_path: str   = None,
        no_refine:       bool  = False,
        branch:          str   = "driving",
        use_fam:         bool  = True,
        use_fm:          bool  = True,
        # driving-branch parameters
        num_db_samples:  int   = Config.NUM_DB_SAMPLES,
        num_db_features: int   = Config.NUM_DB_FEATURES,
        db_hidden_dim:   int   = Config.DB_HIDDEN_DIM,
        db_feat_dim:     int   = Config.DB_FEAT_DIM,
        # body-branch parameters
        num_bg_frames:   int   = Config.NUM_BG_FRAMES,
        num_bg_joints:   int   = Config.NUM_BG_JOINTS,
        use_visibility:  bool  = True,
        use_bones:       bool  = True,
        bg_hidden_dim:   int   = Config.BG_HIDDEN_DIM,
        bg_feat_dim:     int   = Config.BG_FEAT_DIM,
    ):
        super().__init__()
        self.branch = branch
        self.no_refine = no_refine
        self.use_visibility = use_visibility
        self.use_bones = use_bones
        self.use_fam = use_fam
        self.use_fm = use_fm
        if not no_refine and not use_fam:
            raise ValueError("Body refinement requires FAM; disable refinement for FEB ablations")

        self.fefem = FEFEM(pretrained=pretrained, pretrained_path=pretrained_path)
        if use_fm:
            self.fm1 = FM(feat_dim=feat_dim, num_frames=num_frames)
            self.fm2 = FM(feat_dim=feat_dim, num_frames=num_frames)
        if use_fam:
            self.fam = FAM(feat_dim=feat_dim, reduction=fam_reduction)

        if branch == "driving":
            self.num_secondary_samples = num_db_samples
            self.secondary = DBFEM(
                input_dim  = num_db_samples * num_db_features,
                hidden_dim = db_hidden_dim,
                feat_dim   = db_feat_dim,
            )
            secondary_feat_dim = db_feat_dim
        else:  # "body"
            self.num_secondary_samples = num_bg_frames   # unused; kept for symmetry
            self.secondary = BGFEM(
                num_frames=num_bg_frames,
                num_joints=num_bg_joints,
                num_bones=Config.NUM_BG_BONES,
                use_visibility=use_visibility,
                use_bones=use_bones,
                hidden_dim=bg_hidden_dim,
                feat_dim=bg_feat_dim,
            )
            self.register_buffer(
                "body_bone_pairs",
                torch.tensor(Config.BGB_BONE_PAIRS, dtype=torch.long),
                persistent=False,
            )
            secondary_feat_dim = bg_feat_dim

        self.dm = DM(face_feat_dim=feat_dim, db_feat_dim=secondary_feat_dim,
                     num_classes=num_classes, dim_output=dim_output)

    def forward(
        self,
        frame_images:    torch.Tensor,   # (B, k, 1, H, W)
        secondary_data:  torch.Tensor,   # (B, N_db, 7) driving  OR  (B, k, J, 4) body
        body_only:       bool = False,   # if True, zero face_feat before DM (body-branch analysis)
    ):
        B, k, C, H, W = frame_images.shape

        # 1. Per-frame features
        frame_feats = self.fefem(frame_images.view(B * k, C, H, W)).view(B, k, -1)

        # 2-4. Facial branch, matching the standalone FEB ablations.
        if self.use_fm and self.use_fam:
            video_feat = self.fm1(frame_feats)
            _, M_fam, scalar_w = self.fam(frame_feats, video_feat)
            face_feat = self.fm2(M_fam)
        elif self.use_fm:
            face_feat = self.fm1(frame_feats)
            scalar_w = torch.ones(B, k, device=frame_feats.device, dtype=frame_feats.dtype)
        else:
            face_feat = frame_feats.mean(dim=1)
            scalar_w = torch.ones(B, k, device=frame_feats.device, dtype=frame_feats.dtype)
        if body_only:
            face_feat = torch.zeros_like(face_feat)

        # 5. Secondary branch refinement + encoding
        if self.branch == "driving":
            # Upsample FAM weights k->N_db, then apply as soft continuous mask.
            if self.no_refine:
                refined = secondary_data
            else:
                w_up = F.interpolate(
                    scalar_w.unsqueeze(1), size=self.num_secondary_samples,
                    mode="linear", align_corners=False,
                ).squeeze(1)                                              # (B, N_db) in (0,1)
                refined = secondary_data * w_up.unsqueeze(-1)             # (B, N_db, 7)
        else:  # "body"
            # Construct all ablation inputs from the same stored (x,y,z,visible)
            # tensor. Facial attention refines geometric channels only.
            xyz = secondary_data[..., :3]
            visibility = secondary_data[..., 3:4]
            if not self.no_refine:
                xyz = xyz * scalar_w[:, :, None, None]

            body_features = [xyz.reshape(B, k, -1)]
            if self.use_visibility:
                body_features.append(visibility.reshape(B, k, -1))
            if self.use_bones:
                first = self.body_bone_pairs[:, 0]
                second = self.body_bone_pairs[:, 1]
                bone_visible = visibility[:, :, first] * visibility[:, :, second]
                bones = (xyz[:, :, second] - xyz[:, :, first]) * bone_visible
                body_features.append(bones.reshape(B, k, -1))
            refined = torch.cat(body_features, dim=-1)

        # 6. Encode secondary branch
        secondary_feat = self.secondary(refined.reshape(B, -1))

        # 7. Classify
        discrete_logits, dim_pred = self.dm(face_feat, secondary_feat)
        return discrete_logits, dim_pred, scalar_w


# ---------------------------------------------------------------------------
# FacialExpressionBranch  (ablation variant)
# ---------------------------------------------------------------------------

class FacialExpressionBranch(nn.Module):
    """
    FEFEM + FM + FAM + FM for facial-expression-only emotion recognition.
    Used in ablation experiments (FEB, FEB w/o FAM, FEB w/o FAM/FM).
    """

    def __init__(
        self,
        num_frames:    int  = Config.NUM_FRAMES,
        feat_dim:      int  = Config.FEAT_DIM,
        fam_reduction: int  = Config.FAM_REDUCTION,
        num_classes:   int  = Config.NUM_CLASSES,
        dim_output:    int  = Config.DIM_OUTPUT,
        use_fam:       bool = True,
        use_fm:        bool = True,
        pretrained:    bool = True,
        pretrained_path: str = None,
    ):
        super().__init__()
        self.use_fam = use_fam
        self.use_fm  = use_fm

        self.fefem = FEFEM(pretrained=pretrained, pretrained_path=pretrained_path)
        if use_fm:
            self.fm1 = FM(feat_dim=feat_dim, num_frames=num_frames)
            self.fm2 = FM(feat_dim=feat_dim, num_frames=num_frames)
        if use_fam:
            self.fam = FAM(feat_dim=feat_dim, reduction=fam_reduction)

        self.discrete_head    = nn.Linear(feat_dim, num_classes)
        self.dimensional_head = nn.Linear(feat_dim, dim_output)

    def forward(self, frame_images: torch.Tensor):
        B, k, C, H, W = frame_images.shape
        frame_feats = self.fefem(frame_images.view(B * k, C, H, W)).view(B, k, -1)

        if self.use_fm and self.use_fam:
            video_feat = self.fm1(frame_feats)
            _, M_fam, _ = self.fam(frame_feats, video_feat)
            face_feat   = self.fm2(M_fam)
        elif self.use_fm:
            face_feat = self.fm1(frame_feats)
        else:
            face_feat = frame_feats.mean(dim=1)

        dim_out = torch.sigmoid(self.dimensional_head(face_feat)) * 8 + 1
        return self.discrete_head(face_feat), dim_out
