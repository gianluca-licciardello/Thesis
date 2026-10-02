"""
ablation_aide.py - Ablation study for the Body Gesture Branch (BGB) on AIDE.

Experiments (mirroring ablation.py but with branch='body' for MDERNet variants):
  FEB              - FacialExpressionBranch (5-class, AIDE face only)
  FEB (w/o FAM)    - FacialExpressionBranch without FAM
  FEB (w/o FAM/FM) - FacialExpressionBranch without FAM and FM
  BGB (w/o refine)                  - independent coordinates + visibility + bones
  BGB (w/o refine/visibility)       - independent coordinates + bones
  MDERNet (FEB + BGB)                - full inputs with face refinement
  MDERNet (FEB + BGB w/o refine)     - full inputs without face refinement
  Selected FEB + BGB                 - trained middle-level fusion of best branches

AIDE has no VAD (dimensional) labels, so lambda_mse=0 and lambda_ccc=0.
Evaluation still reports MSE/CCC (comparing predictions to zero targets)
but these are not meaningful for AIDE and are not used during training.

Dataset subsets
---------------
  --subset full      : all 2,898 preprocessed clips (2,608.2 train/fold average)
  --subset balanced  : 1,464 retained clips (1,317.6 train/fold average)
  --subset clean     : 2,316 retained clips (2,084.4 train/fold average)

Usage
-----
  python ablation_aide.py --subset full     --all_folds
  python ablation_aide.py --subset balanced --all_folds
  python ablation_aide.py --subset clean    --all_folds --run_dir outputs/aide_clean
"""

import os
import csv
import json
import argparse
import logging
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from config      import Config
from model       import MDERNet, FacialExpressionBranch
from dataset_aide import AIDEDataset, build_kfold_splits_aide
from evaluate    import (combined_loss, evaluate_model, evaluate_model_feb,
                         compute_accuracy, compute_macro_accuracy, compute_f1, compute_weighted_f1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EMOTION_LABELS = Config.AIDE_EMOTION_NAMES   # 5 classes
NUM_CLASSES    = Config.AIDE_NUM_CLASSES

# AIDE has no VAD — dimensional loss is disabled
LAMBDA_MSE = 0.0
LAMBDA_CCC = 0.0
# Soft macro F1 loss compensates for class imbalance on full/clean AIDE subsets
LAMBDA_F1  = Config.LAMBDA_F1

ABLATION_DIR = os.path.join(Config.OUTPUT_DIR, "ablation_aide")

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

# (key, display_name, branch_type, model_kwargs)
VARIANTS = [
    ("feb",            "FEB",                          "feb",      dict(use_fam=True,  use_fm=True)),
    ("feb_no_fam",     "FEB (w/o FAM)",                "feb",      dict(use_fam=False, use_fm=True)),
    ("feb_no_fam_fm",  "FEB (w/o FAM/FM)",             "feb",      dict(use_fam=False, use_fm=False)),
    ("bgb_only_nr",   "BGB (w/o refine)",                    "bgb_only", dict(no_refine=True, use_visibility=True,  use_bones=True)),
    ("bgb_only_nr_nv","BGB (w/o refine/visibility)",         "bgb_only", dict(no_refine=True, use_visibility=False, use_bones=True)),
    ("mdernet_bgb",   "MDERNet (FEB + BGB)",                  "mder",     dict(no_refine=False, use_visibility=True, use_bones=True)),
    ("mdernet_bgb_nr","MDERNet (FEB + BGB w/o refine)",      "mder",     dict(no_refine=True, use_visibility=True, use_bones=True)),
]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_loaders(df, train_idx, test_idx, batch_size, preproc_dir):
    train_ds = AIDEDataset(df, train_idx, preproc_dir=preproc_dir)
    test_ds  = AIDEDataset(df, test_idx, preproc_dir=preproc_dir)
    kw = dict(num_workers=Config.NUM_WORKERS, pin_memory=(Config.DEVICE == "cuda"))
    drop_last = len(train_idx) > batch_size
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=drop_last, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


def _make_opt_sched(model, T_0=None):
    opt = optim.SGD(
        model.parameters(), lr=Config.LR,
        momentum=Config.MOMENTUM, nesterov=Config.NESTEROV,
        weight_decay=Config.WEIGHT_DECAY,
    )
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=(T_0 if T_0 is not None else Config.T_0), eta_min=1e-5)
    return opt, sched


def _train_feb_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples, n = 0.0, 0, 0, 0
    for faces, _body, disc_labels, dim_labels in loader:
        faces       = faces.to(device)
        disc_labels = disc_labels.to(device)
        dim_labels  = dim_labels.to(device)
        optimizer.zero_grad()
        logits, dim_pred = model(faces)
        loss, _ = combined_loss(logits, dim_pred, disc_labels, dim_labels,
                                LAMBDA_MSE, LAMBDA_CCC, LAMBDA_F1)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        total_correct += (logits.argmax(dim=1) == disc_labels).sum().item()
        total_samples += disc_labels.size(0)
        n += 1
    return total_loss / max(n, 1), total_correct / max(total_samples, 1)


def _train_mder_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples, n = 0.0, 0, 0, 0
    for faces, body, disc_labels, dim_labels in loader:
        faces       = faces.to(device)
        body        = body.to(device)
        disc_labels = disc_labels.to(device)
        dim_labels  = dim_labels.to(device)
        optimizer.zero_grad()
        logits, dim_pred, _ = model(faces, body)
        loss, _ = combined_loss(logits, dim_pred, disc_labels, dim_labels,
                                LAMBDA_MSE, LAMBDA_CCC, LAMBDA_F1)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        total_correct += (logits.argmax(dim=1) == disc_labels).sum().item()
        total_samples += disc_labels.size(0)
        n += 1
    return total_loss / max(n, 1), total_correct / max(total_samples, 1)


def _train_bgb_only_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples, n = 0.0, 0, 0, 0
    for faces, body, disc_labels, dim_labels in loader:
        faces       = faces.to(device)
        body        = body.to(device)
        disc_labels = disc_labels.to(device)
        dim_labels  = dim_labels.to(device)
        optimizer.zero_grad()
        logits, dim_pred, _ = model(faces, body, body_only=True)
        loss, _ = combined_loss(logits, dim_pred, disc_labels, dim_labels,
                                LAMBDA_MSE, LAMBDA_CCC, LAMBDA_F1)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        total_correct += (logits.argmax(dim=1) == disc_labels).sum().item()
        total_samples += disc_labels.size(0)
        n += 1
    return total_loss / max(n, 1), total_correct / max(total_samples, 1)


def save_fold_curves(history, variant_key, fold_id,
                     disc_true=None, disc_pred=None,
                     test_acc=None, test_f1=None):
    curves_dir = os.path.join(ABLATION_DIR, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    epochs    = [h["epoch"]          for h in history]
    tr_loss   = [h["train_loss"]     for h in history]
    val_loss  = [h.get("val_loss")   for h in history]
    val_acc   = [h["accuracy"] * 100 for h in history]
    train_acc = [h.get("train_acc")  for h in history]

    has_val_loss  = all(v is not None for v in val_loss)
    has_train_acc = all(v is not None for v in train_acc)
    has_cm        = disc_true is not None and disc_pred is not None
    ncols         = 3 if has_cm else 2

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(f"{variant_key}  fold {fold_id:02d}", fontsize=12)

    axes[0].plot(epochs, tr_loss, marker="o", markersize=3, label="Train loss")
    if has_val_loss:
        axes[0].plot(epochs, val_loss, marker="o", markersize=3,
                     color="tab:red", label="Val loss")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_acc, marker="o", markersize=3,
                 color="tab:orange", label="Val acc")
    if has_train_acc:
        axes[1].plot(epochs, [v * 100 for v in train_acc], marker="o", markersize=3,
                     color="tab:blue", label="Train acc")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy"); axes[1].set_ylim(0, 100); axes[1].grid(True, alpha=0.3)

    best_val = max(val_acc) if val_acc else 0.0
    lines = [f"Best val: {best_val:.2f}%"]
    if test_acc is not None:
        lines.append(f"Test acc: {test_acc * 100:.2f}%")
    if test_f1 is not None:
        lines.append(f"Test F1:  {test_f1 * 100:.2f}%")
    axes[1].text(0.97, 0.05, "\n".join(lines), transform=axes[1].transAxes,
                 fontsize=9, va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    if has_cm:
        short = [n[:3] for n in EMOTION_LABELS]
        cm = confusion_matrix(disc_true, disc_pred,
                              labels=list(range(NUM_CLASSES)), normalize="true")
        sns.heatmap(cm, ax=axes[2], annot=True, fmt=".2f",
                    cmap="YlOrRd", vmin=0, vmax=1.0,
                    xticklabels=short, yticklabels=short,
                    linewidths=0.5, linecolor="lightgray", cbar=False)
        axes[2].set_xlabel("Predicted", fontsize=9)
        axes[2].set_ylabel("True", fontsize=9)
        axes[2].set_title("Test confusion matrix")
        axes[2].tick_params(axis="x", rotation=45, labelsize=8)
        axes[2].tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    path = os.path.join(curves_dir, f"{variant_key}_fold_{fold_id:02d}_curves.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    log.info("  Saved curves -> %s", path)


def train_fold(variant_key, fold_id, df, train_idx, test_idx, args, device, variant_spec=None):
    """Train one fold; returns (metrics_dict, disc_true, disc_pred, history)."""
    _, display_name, branch_type, model_kwargs = variant_spec or next(
        v for v in VARIANTS if v[0] == variant_key
    )

    train_loader, test_loader = _make_loaders(df, train_idx, test_idx, args.batch_size, args.preproc_dir)

    if branch_type == "feb":
        model = FacialExpressionBranch(
            num_classes     = NUM_CLASSES,
            pretrained      = args.pretrained,
            pretrained_path = args.pretrained_path,
            **model_kwargs,
        ).to(device)
    else:  # "mder" or "bgb_only"
        model = MDERNet(
            num_classes     = NUM_CLASSES,
            branch          = "body",
            pretrained      = args.pretrained,
            pretrained_path = args.pretrained_path,
            **model_kwargs,
        ).to(device)

    body_only = (branch_type == "bgb_only")

    optimizer, scheduler = _make_opt_sched(model, T_0=getattr(args, "T_0", None))
    best_acc   = 0.0
    best_state = None
    history    = []

    for epoch in range(1, args.epochs + 1):
        if branch_type == "feb":
            avg_loss, train_acc = _train_feb_epoch(model, train_loader, optimizer, device)
            metrics = evaluate_model_feb(model, test_loader, device,
                                         lambda_mse=LAMBDA_MSE, lambda_ccc=LAMBDA_CCC,
                                         lambda_f1=LAMBDA_F1)
        elif body_only:
            avg_loss, train_acc = _train_bgb_only_epoch(model, train_loader, optimizer, device)
            metrics = evaluate_model(model, test_loader, device,
                                     lambda_mse=LAMBDA_MSE, lambda_ccc=LAMBDA_CCC,
                                     lambda_f1=LAMBDA_F1, body_only=True)
        else:
            avg_loss, train_acc = _train_mder_epoch(model, train_loader, optimizer, device)
            metrics = evaluate_model(model, test_loader, device,
                                     lambda_mse=LAMBDA_MSE, lambda_ccc=LAMBDA_CCC,
                                     lambda_f1=LAMBDA_F1)

        scheduler.step()
        history.append({"epoch": epoch, "train_loss": avg_loss,
                        "train_acc": train_acc, **metrics})

        if metrics["accuracy"] > best_acc:
            best_acc   = metrics["accuracy"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == args.epochs:
            log.info("  [%s] fold=%d  epoch=%3d/%d  loss=%.4f  acc=%.4f",
                     display_name, fold_id, epoch, args.epochs,
                     avg_loss, metrics["accuracy"])

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if branch_type == "feb":
        final_metrics, disc_true, disc_pred, disc_probs = evaluate_model_feb(
            model, test_loader, device,
            lambda_mse=LAMBDA_MSE, lambda_ccc=LAMBDA_CCC, lambda_f1=LAMBDA_F1,
            return_preds=True, return_probs=True)
    else:
        final_metrics, disc_true, disc_pred, disc_probs = evaluate_model(
            model, test_loader, device,
            lambda_mse=LAMBDA_MSE, lambda_ccc=LAMBDA_CCC, lambda_f1=LAMBDA_F1,
            return_preds=True, return_probs=True, body_only=body_only)

    return final_metrics, disc_true, disc_pred, disc_probs, history


# ---------------------------------------------------------------------------
# Run all variants
# ---------------------------------------------------------------------------

def run_variants(df, splits, args, device, results):
    fold_ids = sorted(set(args.fold_subset if args.fold_subset else range(args.folds)))

    for key, name, branch_type, _ in VARIANTS:
        if args.feb_only and branch_type != "feb":
            continue
        if args.mder_only and branch_type not in ("mder", "bgb_only"):
            continue
        if key in results:
            body_variant = branch_type in ("mder", "bgb_only")
            if body_variant and results[key].get("body_schema") != Config.BGB_SCHEMA:
                log.info("Discarding stale %-30s cache (body schema changed)", name)
                del results[key]
            else:
                log.info("Skipping %-30s (cached in results.json)", name)
                continue

        log.info("=" * 70)
        log.info("VARIANT: %s", name)
        log.info("=" * 70)

        fold_metrics = []
        all_true, all_pred = [], []

        for fold_id in fold_ids:
            train_idx, test_idx = splits[fold_id]
            t0 = time.time()
            metrics, disc_true, disc_pred, disc_probs, history = train_fold(
                key, fold_id, df, train_idx, test_idx, args, device
            )
            save_fold_curves(history, key, fold_id,
                             disc_true=disc_true, disc_pred=disc_pred,
                             test_acc=metrics["accuracy"], test_f1=metrics["f1"])
            elapsed = time.time() - t0
            log.info("  fold=%d done in %.1fs  acc=%.4f  f1=%.4f",
                     fold_id, elapsed, metrics["accuracy"], metrics["f1"])
            metrics["macro_f1"] = compute_f1(disc_true, disc_pred)
            metrics["f1"] = compute_weighted_f1(disc_true, disc_pred)
            fold_metrics.append({
                **metrics, "fold": fold_id,
                "disc_true":  disc_true.tolist(),
                "disc_pred":  disc_pred.tolist(),
                "disc_probs": disc_probs.tolist(),
                "history":    history,
            })
            all_true.extend(disc_true.tolist())
            all_pred.extend(disc_pred.tolist())

        results[key] = {
            "name":      name,
            "type":      branch_type,
            **({"body_schema": Config.BGB_SCHEMA} if branch_type != "feb" else {}),
            "accuracy":       float(np.mean([m["accuracy"]       for m in fold_metrics])),
            "macro_accuracy": float(np.mean([m["macro_accuracy"] for m in fold_metrics])),
            "f1":             float(np.mean([m["f1"]             for m in fold_metrics])),
            "macro_f1":       float(np.mean([m["macro_f1"]       for m in fold_metrics])),
            "n_folds":        len(fold_metrics),
            "folds":     fold_metrics,
            "disc_true": all_true,
            "disc_pred": all_pred,
        }
        _save_results(results)
        log.info("  --> mean: acc=%.4f  f1=%.4f",
                 results[key]["accuracy"], results[key]["f1"])

    return results


# ---------------------------------------------------------------------------
# Selected-branch middle-level fusion
# ---------------------------------------------------------------------------

def run_combined_analysis(results, df=None, splits=None, args=None, device=None):
    """Train/reuse MDERNet configured from the best standalone FEB and BGB."""
    feb_keys = ["feb", "feb_no_fam", "feb_no_fam_fm"]
    bgb_keys = ["bgb_only_nr", "bgb_only_nr_nv"]
    if not all(k in results for k in feb_keys + bgb_keys):
        log.warning("Selected-branch MDERNet skipped: standalone branch results are incomplete.")
        return results
    best_feb = max(feb_keys, key=lambda k: results[k]["accuracy"])
    best_bgb = max(bgb_keys, key=lambda k: results[k]["accuracy"])
    specs = {k: kw for k, _n, _t, kw in VARIANTS}
    kwargs = {**specs[best_feb], **specs[best_bgb]}
    name = f"MDERNet ({results[best_feb]['name']} + {results[best_bgb]['name']})"
    signature = f"{best_feb}+{best_bgb}+{Config.BGB_SCHEMA}"
    # Copy an identical fixed result into the selected row; do not retrain it.
    if best_feb == "feb" and best_bgb == "bgb_only_nr" and "mdernet_bgb_nr" in results:
        selected = json.loads(json.dumps(results["mdernet_bgb_nr"]))
        selected.update(name=name, type="selected_mdernet", feb_variant=best_feb,
                        bgb_variant=best_bgb, selection_signature=signature,
                        source_variant="mdernet_bgb_nr")
        results["combined_best"] = selected
        _save_results(results)
        return results
    cached = results.get("combined_best", {})
    if cached.get("selection_signature") == signature:
        return results
    if df is None:
        results.pop("combined_best", None)
        log.warning("Selected architecture %s requires training; figures-only mode cannot train it.", name)
        return results

    spec = ("combined_best", name, "mder", kwargs)
    fold_ids = sorted(set(args.fold_subset if args.fold_subset else range(args.folds)))
    fold_metrics, all_true, all_pred = [], [], []
    for fold_id in fold_ids:
        train_idx, test_idx = splits[fold_id]
        metrics, disc_true, disc_pred, disc_probs, history = train_fold(
            "combined_best", fold_id, df, train_idx, test_idx, args, device, variant_spec=spec)
        metrics["macro_f1"] = compute_f1(disc_true, disc_pred)
        metrics["f1"] = compute_weighted_f1(disc_true, disc_pred)
        save_fold_curves(history, "combined_best", fold_id, disc_true, disc_pred,
                         metrics["accuracy"], metrics["f1"])
        fold_metrics.append({**metrics, "fold": fold_id, "disc_true": disc_true.tolist(),
                             "disc_pred": disc_pred.tolist(), "disc_probs": disc_probs.tolist(),
                             "history": history})
        all_true.extend(disc_true.tolist()); all_pred.extend(disc_pred.tolist())
    results["combined_best"] = {
        "name": name, "type": "selected_mdernet", "body_schema": Config.BGB_SCHEMA,
        "feb_variant": best_feb, "bgb_variant": best_bgb,
        "selection_signature": signature,
        "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "macro_accuracy": float(np.mean([m["macro_accuracy"] for m in fold_metrics])),
        "f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "n_folds": len(fold_metrics), "folds": fold_metrics,
        "disc_true": all_true, "disc_pred": all_pred,
    }
    _save_results(results)
    return results


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def ensure_macro_metrics(results):
    """Backfill macro accuracy, weighted F1, and macro F1 from cached predictions."""
    changed = False
    for result in results.values():
        folds = result.get("folds", [])
        for fold in folds:
            y_true, y_pred = fold.get("disc_true"), fold.get("disc_pred")
            if y_true is None or y_pred is None:
                continue
            y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
            values = {
                "macro_accuracy": compute_macro_accuracy(y_true, y_pred),
                "f1": compute_weighted_f1(y_true, y_pred),
                "macro_f1": compute_f1(y_true, y_pred),
            }
            for key, value in values.items():
                if fold.get(key) != value:
                    fold[key] = value
                    changed = True
        if folds and all(all(k in fold for k in ("macro_accuracy", "f1", "macro_f1")) for fold in folds):
            for key in ("macro_accuracy", "f1", "macro_f1"):
                value = float(np.mean([fold[key] for fold in folds]))
                if result.get(key) != value:
                    result[key] = value
                    changed = True
    return changed


def _results_path():
    return os.path.join(ABLATION_DIR, "results.json")


def _load_results():
    path = _results_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_results(results):
    os.makedirs(ABLATION_DIR, exist_ok=True)
    with open(_results_path(), "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def _print_table(title, headers, rows):
    widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "|" + "|".join(f" {{:<{w}}} " for w in widths) + "|"
    print(f"\n{title}")
    print(sep); print(fmt.format(*headers)); print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


def _save_csv(filename, headers, rows):
    path = os.path.join(ABLATION_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    log.info("Saved %s", path)


def build_tables(results):
    feb_keys      = ["feb", "feb_no_fam", "feb_no_fam_fm"]
    mder_keys     = ["mdernet_bgb", "mdernet_bgb_nr", "combined_best"]
    bgb_only_keys = ["bgb_only_nr", "bgb_only_nr_nv"]

    def fmt(r):
        return [r["name"], f"{r['accuracy']*100:.2f}%", f"{r['f1']*100:.2f}%",
                f"{r['macro_accuracy']*100:.2f}%", f"{r['macro_f1']*100:.2f}%"]

    table_feb      = [fmt(results[k]) for k in feb_keys      if k in results]
    table_mder     = [fmt(results[k]) for k in mder_keys     if k in results]
    table_bgb_only = [fmt(results[k]) for k in bgb_only_keys if k in results]

    return table_feb, table_mder, table_bgb_only


def print_tables(results):
    headers = ["Models", "Accuracy", "F1 score", "Macro accuracy", "Macro F1 score"]
    t_feb, t_mder, t_bgb = build_tables(results)
    if t_feb:
        _print_table("FEB variants (AIDE)", headers, t_feb)
    if t_mder:
        _print_table("MDERNet-BGB variants (AIDE)", headers, t_mder)
    if t_bgb:
        _print_table("BGB-only variants (AIDE)", headers, t_bgb)


def save_csv_tables(results):
    headers = ["Models", "Accuracy", "F1 score", "Macro accuracy", "Macro F1 score"]
    t_feb, t_mder, t_bgb = build_tables(results)
    if t_feb:
        _save_csv("table_feb.csv", headers, t_feb)
    if t_mder:
        _save_csv("table_mder.csv", headers, t_mder)
    if t_bgb:
        _save_csv("table_bgb_only.csv", headers, t_bgb)


def save_table_figure(results):
    headers = ["Models", "Accuracy", "F1 score", "Macro accuracy", "Macro F1 score"]

    ORDERED = [
        ("feb",            "#DDEEFF"),
        ("feb_no_fam",     "#DDEEFF"),
        ("feb_no_fam_fm",  "#DDEEFF"),
        ("bgb_only_nr",    "#DDEEDD"),
        ("bgb_only_nr_nv", "#DDEEDD"),
        ("mdernet_bgb",    "#FFEEDD"),

        ("mdernet_bgb_nr", "#FFEEDD"),
        ("combined_best",  "#F0E0FF"),
    ]

    def fmt(r):
        return [r["name"], f"{r['accuracy']*100:.2f}%", f"{r['f1']*100:.2f}%",
                f"{r['macro_accuracy']*100:.2f}%", f"{r['macro_f1']*100:.2f}%"]

    rows, row_colors = [], []
    for key, color in ORDERED:
        if key in results:
            rows.append(fmt(results[key]))
            row_colors.append(color)

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.52 * len(rows)))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, color in enumerate(row_colors, start=1):
        for j in range(len(headers)):
            tbl[i, j].set_facecolor(color)

    ax.set_title("Ablation results (AIDE)", fontsize=10, fontweight="bold", loc="left", pad=6)
    plt.tight_layout()
    path = os.path.join(ABLATION_DIR, "tables.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved tables figure -> %s", path)


def _save_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)), normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0.0, vmax=1.0,
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                linewidths=0.5, linecolor="lightgray", cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted labels", fontsize=11)
    ax.set_ylabel("True labels", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    path = os.path.join(ABLATION_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix -> %s", path)


def save_confusion_matrix(results):
    if "mdernet_bgb" in results:
        r = results["mdernet_bgb"]
        _save_cm(np.array(r["disc_true"]), np.array(r["disc_pred"]),
                 "Confusion matrix  (MDERNet-BGB · AIDE)", "confusion_matrix.png")
    else:
        log.warning("MDERNet-BGB results not found; skipping confusion matrix.")

    if "combined_best" in results:
        r = results["combined_best"]
        _save_cm(np.array(r["disc_true"]), np.array(r["disc_pred"]),
                 f"Confusion matrix  ({r['name']} · AIDE)",
                 "confusion_matrix_combined.png")


def save_best_fold_analysis(results):
    best = {}
    for key, name, _, _ in VARIANTS:
        if key not in results:
            continue
        folds = results[key].get("folds", [])
        if not folds:
            continue
        bf = max(folds, key=lambda f: f["accuracy"])
        best[key] = {
            "name": name, "fold": bf["fold"],
            "accuracy": bf["accuracy"], "f1": bf["f1"],
            "macro_accuracy": bf["macro_accuracy"], "macro_f1": bf["macro_f1"],
            "disc_true": bf.get("disc_true"), "disc_pred": bf.get("disc_pred"),
        }

    if not best:
        return

    for key, data in best.items():
        if data["disc_true"] is None:
            continue
        y_true = np.array(data["disc_true"])
        y_pred = np.array(data["disc_pred"])
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)), normalize="true")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1.0,
                    xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                    linewidths=0.5, linecolor="lightgray", cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Predicted labels", fontsize=11)
        ax.set_ylabel("True labels", fontsize=11)
        ax.set_title(
            f"Best fold — {data['name']}  (fold {data['fold']:02d}  "
            f"acc={data['accuracy']*100:.2f}%)", fontsize=11)
        plt.tight_layout()
        path = os.path.join(ABLATION_DIR, f"best_fold_cm_{key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved best-fold CM -> %s", path)

    rows = []
    for key, name, _, _ in VARIANTS:
        if key not in best:
            continue
        d = best[key]
        rows.append([d["name"], str(d["fold"]), f"{d['accuracy']*100:.2f}%",
                     f"{d['f1']*100:.2f}%", f"{d['macro_accuracy']*100:.2f}%",
                     f"{d['macro_f1']*100:.2f}%"])

    if not rows:
        return
    headers = ["Model", "Best fold", "Accuracy", "F1 score", "Macro accuracy", "Macro F1 score"]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        c = "#EEF2FF" if i % 2 == 1 else "white"
        for j in range(len(headers)):
            tbl[i, j].set_facecolor(c)
    ax.set_title("Best-fold results per variant (AIDE)", fontsize=10,
                 fontweight="bold", loc="left", pad=4)
    plt.tight_layout()
    path = os.path.join(ABLATION_DIR, "best_fold_tables.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved best-fold table -> %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation study: MDERNet-BGB on AIDE")
    parser.add_argument("--subset",         type=str, default="full",
                        choices=["full", "balanced", "clean"],
                        help="AIDE subset to use for training and evaluation.")
    parser.add_argument("--epochs",         type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--folds",          type=int, default=Config.K_FOLDS)
    parser.add_argument("--fold_id",        type=int, default=0)
    parser.add_argument("--all_folds",      action="store_true")
    parser.add_argument("--fold_subset",    type=int, nargs="+", default=None)
    parser.add_argument("--feb_only",       action="store_true")
    parser.add_argument("--mder_only",      action="store_true")
    parser.add_argument("--pretrained",     action="store_true", default=True)
    parser.add_argument("--no_pretrained",  dest="pretrained", action="store_false")
    parser.add_argument("--pretrained_path",type=str, default=None)
    parser.add_argument("--batch_size",     type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--T_0",           type=int, default=Config.T_0)
    parser.add_argument("--device",         type=str, default=Config.DEVICE)
    parser.add_argument("--preproc_dir",    type=str, default=Config.AIDE_PREPROCESSED_DIR)
    parser.add_argument("--figures_only",   action="store_true")
    parser.add_argument("--run_dir",        type=str, default=None)
    args = parser.parse_args()

    global ABLATION_DIR
    Config.ensure_dirs()

    subset_tag = args.subset

    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
        ABLATION_DIR = os.path.join(run_dir, "ablation")
        os.makedirs(ABLATION_DIR, exist_ok=True)
    elif args.figures_only:
        latest = Config.latest_run_dir(Config.OUTPUT_DIR)
        if latest is None:
            log.error("No run_N directories found under %s.", Config.OUTPUT_DIR)
            return
        ABLATION_DIR = os.path.join(latest, "ablation")
        log.info("--figures_only: loading from %s", ABLATION_DIR)
    else:
        run_dir = Config.next_run_dir(Config.OUTPUT_DIR)
        ABLATION_DIR = os.path.join(run_dir, f"ablation_aide_{subset_tag}")
        os.makedirs(ABLATION_DIR, exist_ok=True)
        log.info("Run dir: %s", run_dir)

    log.info("Device:  %s", args.device)
    log.info("Subset:  %s", subset_tag)
    log.info("Output:  %s", ABLATION_DIR)

    results = _load_results()
    if results:
        if results.pop("bgb_only", None) is not None:
            _save_results(results)
            log.info("Removed obsolete face-guided BGB-only cached result.")
        log.info("Loaded %d cached variant(s) from results.json", len(results))
        if ensure_macro_metrics(results):
            _save_results(results)
            log.info("Backfilled macro metrics from cached predictions.")

    if not args.figures_only:
        subset_arg = None if args.subset == "full" else args.subset
        df, splits = build_kfold_splits_aide(
            preproc_dir=args.preproc_dir, k_folds=args.folds, subset=subset_arg
        )
        log.info("Dataset: %d samples | %d folds | subset=%s", len(df), len(splits), subset_tag)

        if args.fold_subset is None and not args.all_folds:
            args.fold_subset = [args.fold_id]

        results = run_variants(df, splits, args, args.device, results)
        results = run_combined_analysis(results, df, splits, args, args.device)

    if args.figures_only:
        for key, name, _, _ in VARIANTS:
            if key not in results:
                continue
            for fold_data in results[key].get("folds", []):
                hist = fold_data.get("history")
                if not hist:
                    continue
                dt = fold_data.get("disc_true")
                dp = fold_data.get("disc_pred")
                save_fold_curves(
                    hist, key, fold_data["fold"],
                    disc_true=np.array(dt) if dt is not None else None,
                    disc_pred=np.array(dp) if dp is not None else None,
                    test_acc=fold_data.get("accuracy"),
                    test_f1=fold_data.get("f1"),
                )

    if args.figures_only:
        results = run_combined_analysis(results)

    print_tables(results)
    save_csv_tables(results)
    save_table_figure(results)
    save_confusion_matrix(results)
    save_best_fold_analysis(results)

    log.info("All outputs saved to %s", ABLATION_DIR)


if __name__ == "__main__":
    main()
