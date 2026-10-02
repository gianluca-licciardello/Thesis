"""
evaluate.py – Loss functions and evaluation metrics for MDERNet.

Metrics (paper §4.3.2)
-----------------------
Discrete emotion (classification):
  • Accuracy
  • Macro accuracy (balanced accuracy; mean per-class recall)
  • F1-score (macro-averaged)

Dimensional emotion (regression):
  • MSE  – mean square error  (lower is better)
  • CCC  – concordance correlation coefficient  (higher is better, range −1…1)

Loss functions (paper §3.5)
----------------------------
  L_cross_entropy  – for discrete classification (optimises Accuracy)
  L_MSE            – for dimensional regression
  L_CCC            – for dimensional regression (encourages mean + variance match)

The combined training loss is:
  L_total = L_cross_entropy + λ_MSE * L_MSE + λ_CCC * L_CCC
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Concordance Correlation Coefficient loss (paper Eq. 10).

      L_CCC = 1 − mean_over_dims( 2*cov / (var_pred + var_target + (μ_pred−μ_target)²) )

    Works on a batch: pred/target are (B, D) tensors.
    Returns a scalar loss ∈ [0, 2]  (0 = perfect prediction).
    """
    pred_mean   = pred.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_var    = pred.var(dim=0, unbiased=False)
    target_var  = target.var(dim=0, unbiased=False)
    cov         = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)

    ccc = (2 * cov) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + eps)
    return (1 - ccc).mean()


def soft_macro_f1_loss(
    logits: torch.Tensor,   # (B, C)
    labels: torch.Tensor,   # (B,)
    eps:    float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable macro F1 loss: 1 − macro_F1, in [0, 1].

    Uses soft (probability) predictions so the gradient flows through the
    softmax layer.  Per-class soft TP, FP, FN are computed from the
    probability mass on each class, then averaged over all C classes.

    Optimising this term alongside cross-entropy compensates for class
    imbalance: cross-entropy weights samples equally, while soft macro F1
    gives each class equal influence on the gradient regardless of its size.
    """
    num_classes = logits.shape[-1]
    probs   = torch.softmax(logits, dim=1)               # (B, C)
    one_hot = F.one_hot(labels, num_classes).float()     # (B, C)

    tp = (probs * one_hot).sum(dim=0)                    # (C,)
    fp = (probs * (1 - one_hot)).sum(dim=0)              # (C,)
    fn = ((1 - probs) * one_hot).sum(dim=0)              # (C,)

    f1_per_class = 2 * tp / (2 * tp + fp + fn + eps)    # (C,)
    return 1 - f1_per_class.mean()


def combined_loss(
    discrete_logits: torch.Tensor,   # (B, C)
    dim_pred:        torch.Tensor,   # (B, 3)
    discrete_labels: torch.Tensor,   # (B,)
    dim_labels:      torch.Tensor,   # (B, 3)
    lambda_mse:      float = 0.5,
    lambda_ccc:      float = 0.5,
    lambda_f1:       float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss used for training MDERNet.

    lambda_f1 > 0 adds a soft macro F1 term — used for AIDE (no VAD labels)
    to compensate for class imbalance.  Defaults to 0 so PPB-EMO training
    is unchanged.

    Returns
    -------
    total_loss : scalar Tensor
    components : dict with keys 'ce', 'mse', 'ccc', and optionally 'f1'
    """
    l_ce  = F.cross_entropy(discrete_logits, discrete_labels)
    l_mse = F.mse_loss(dim_pred, dim_labels)
    l_ccc = ccc_loss(dim_pred, dim_labels)

    total      = l_ce + lambda_mse * l_mse + lambda_ccc * l_ccc
    components = {"ce": l_ce.item(), "mse": l_mse.item(), "ccc": l_ccc.item()}

    if lambda_f1 > 0.0:
        l_f1 = soft_macro_f1_loss(discrete_logits, discrete_labels)
        total = total + lambda_f1 * l_f1
        components["f1"] = l_f1.item()

    return total, components


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics (NumPy, run on collected predictions)
# ──────────────────────────────────────────────────────────────────────────────

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_macro_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall across classes (also known as balanced accuracy)."""
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1; retained as ``f1`` for result-file compatibility."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Support-weighted multiclass F1, used as AIDE's overall F1-score."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean over samples AND dimensions."""
    return float(np.mean((y_true - y_pred) ** 2))


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean CCC over the three dimensional-emotion dimensions.
    Follows paper Eq. 10.
    """
    ccc_vals = []
    for d in range(y_true.shape[1]):
        t = y_true[:, d].astype(np.float64)
        p = y_pred[:, d].astype(np.float64)
        mu_t, mu_p = t.mean(), p.mean()
        var_t = np.var(t, ddof=0)
        var_p = np.var(p, ddof=0)
        cov   = np.mean((t - mu_t) * (p - mu_p))
        ccc   = (2 * cov) / (var_t + var_p + (mu_t - mu_p) ** 2 + 1e-8)
        ccc_vals.append(ccc)
    return float(np.mean(ccc_vals))


def evaluate_model(
    model:        nn.Module,
    dataloader,
    device:       str,
    return_preds: bool = False,
    return_probs: bool = False,
    lambda_mse:   float = 0.1,
    lambda_ccc:   float = 1.0,
    lambda_f1:    float = 0.0,
    body_only:    bool = False,
) -> dict:
    """
    Run inference on a DataLoader and return all four metrics plus val_loss.

    Returns
    -------
    metrics        : dict with keys: accuracy, f1, mse, ccc, val_loss
    (disc_true, disc_pred)            appended when return_preds=True
    (disc_true, disc_pred, disc_probs) when return_preds=True and return_probs=True
    """
    model.eval()
    all_disc_true  = []
    all_disc_pred  = []
    all_disc_probs = []
    all_dim_true   = []
    all_dim_pred   = []
    total_loss, n  = 0.0, 0

    with torch.no_grad():
        for faces, db, disc_labels, dim_labels in dataloader:
            faces       = faces.to(device)
            db          = db.to(device)
            disc_labels = disc_labels.to(device)
            dim_labels  = dim_labels.to(device)

            logits, dim_out, _ = model(faces, db, body_only=body_only)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            loss, _ = combined_loss(logits, dim_out, disc_labels, dim_labels,
                                    lambda_mse, lambda_ccc, lambda_f1)
            total_loss += loss.item()
            n          += 1

            all_disc_true.append(disc_labels.cpu().numpy())
            all_disc_pred.append(preds.cpu().numpy())
            all_disc_probs.append(probs.cpu().numpy())
            all_dim_true.append(dim_labels.cpu().numpy())
            all_dim_pred.append(dim_out.cpu().numpy())

    disc_true  = np.concatenate(all_disc_true)
    disc_pred  = np.concatenate(all_disc_pred)
    disc_probs = np.concatenate(all_disc_probs)
    dim_true   = np.concatenate(all_dim_true)
    dim_pred   = np.concatenate(all_dim_pred)

    metrics = {
        "accuracy":       compute_accuracy(disc_true, disc_pred),
        "macro_accuracy": compute_macro_accuracy(disc_true, disc_pred),
        "f1":             compute_f1(disc_true, disc_pred),
        "macro_f1":       compute_f1(disc_true, disc_pred),
        "mse":            compute_mse(dim_true, dim_pred),
        "ccc":      compute_ccc(dim_true, dim_pred),
        "val_loss": total_loss / max(n, 1),
    }
    if return_preds:
        if return_probs:
            return metrics, disc_true, disc_pred, disc_probs
        return metrics, disc_true, disc_pred
    return metrics


def evaluate_model_feb(
    model:        nn.Module,
    dataloader,
    device:       str,
    return_preds: bool = False,
    return_probs: bool = False,
    lambda_mse:   float = 0.1,
    lambda_ccc:   float = 1.0,
    lambda_f1:    float = 0.0,
) -> dict:
    """
    Evaluate a FacialExpressionBranch model (takes only face frames as input).

    Returns
    -------
    metrics        : dict with keys: accuracy, f1, mse, ccc, val_loss
    (disc_true, disc_pred)             appended when return_preds=True
    (disc_true, disc_pred, disc_probs) when return_preds=True and return_probs=True
    """
    model.eval()
    all_disc_true  = []
    all_disc_pred  = []
    all_disc_probs = []
    all_dim_true   = []
    all_dim_pred   = []
    total_loss, n  = 0.0, 0

    with torch.no_grad():
        for faces, _db, disc_labels, dim_labels in dataloader:
            faces       = faces.to(device)
            disc_labels = disc_labels.to(device)
            dim_labels  = dim_labels.to(device)

            logits, dim_out = model(faces)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            loss, _ = combined_loss(logits, dim_out, disc_labels, dim_labels,
                                    lambda_mse, lambda_ccc, lambda_f1)
            total_loss += loss.item()
            n          += 1

            all_disc_true.append(disc_labels.cpu().numpy())
            all_disc_pred.append(preds.cpu().numpy())
            all_disc_probs.append(probs.cpu().numpy())
            all_dim_true.append(dim_labels.cpu().numpy())
            all_dim_pred.append(dim_out.cpu().numpy())

    disc_true  = np.concatenate(all_disc_true)
    disc_pred  = np.concatenate(all_disc_pred)
    disc_probs = np.concatenate(all_disc_probs)
    dim_true   = np.concatenate(all_dim_true)
    dim_pred   = np.concatenate(all_dim_pred)

    metrics = {
        "accuracy":       compute_accuracy(disc_true, disc_pred),
        "macro_accuracy": compute_macro_accuracy(disc_true, disc_pred),
        "f1":             compute_f1(disc_true, disc_pred),
        "macro_f1":       compute_f1(disc_true, disc_pred),
        "mse":            compute_mse(dim_true, dim_pred),
        "ccc":      compute_ccc(dim_true, dim_pred),
        "val_loss": total_loss / max(n, 1),
    }
    if return_preds:
        if return_probs:
            return metrics, disc_true, disc_pred, disc_probs
        return metrics, disc_true, disc_pred
    return metrics
