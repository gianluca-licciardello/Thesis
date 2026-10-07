#!/usr/bin/env python3
"""
Evaluation script for DECNet on PPB-EMO / AIDE.

Metrics follow the paper (§IV.B):
  Acc-7        — overall top-1 accuracy
  F1-macro     — macro-averaged F1 (= F1-7 in the paper)
  Acc-bal      — balanced accuracy = macro-averaged per-class recall
                  (= "averaged accuracy (Acc)" in the paper)
  F1-wtd       — weighted-averaged F1 (additional metric; not among the paper's four metrics)

All four metrics are reported for each of the three branches:
  Visual (o_v), DB (o_db), Fused (o)   [Table I ablation in the paper]

Outputs:
  eval-log.txt        — per-fold and summary metrics for all three branches
  combined-figure.png — training curves (if available) + three row-normalised
                        confusion matrices: Visual | DB | Fused
  eval-report.md      — Markdown: summary table, per-class breakdowns,
                        classification reports for all three branches

Two modes:

  Single fold
    python test.py --checkpoint ckpt_best.pth \\
                   --test_txt_path annotation/PPB_CIR_V_DB_rgb_112_3s_test_fold_0.txt

  5-fold cross-validation  (matches paper protocol)
    python test.py --checkpoints fold0.pth fold1.pth ... fold4.pth \\
                   --annotation_dir annotation --t 3
"""

import argparse
import json
import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, balanced_accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator

from models.ST_Former import DECNet

# Defaults for PPB-EMO; overridden at runtime via --emotions / --num_classes
EMOTIONS    = ['AD', 'DD', 'FD', 'HD', 'ND', 'SAD', 'SD']
NUM_CLASSES = len(EMOTIONS)


# ---------------------------------------------------------------------------
# RecorderMeter — mirrors train.py exactly so torch.load can deserialize
# checkpoints (both scripts run as __main__; pickle looks up the class there).
# ---------------------------------------------------------------------------

class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch    = total_epoch
        self.current_epoch  = 0
        self.epoch_losses   = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0]   = train_loss * 50   # ×50 matches train.py
        self.epoch_losses[idx, 1]   = val_loss   * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def evaluate_fold(model, loader):
    """
    Run a single forward pass over the test loader.
    Returns (preds_v, preds_db, preds_fused, targets) as numpy int arrays.
    All three branches are evaluated in one pass to match the MT-JL design.
    """
    model.eval()
    preds_v, preds_db, preds_fused, targets = [], [], [], []
    with torch.no_grad():
        for DB_feature, (images, target), _ in loader:
            DB_feature = DB_feature.cuda()
            images     = images.cuda()
            out_v, out_db, out = model((images, DB_feature), contrast=getattr(model, 'contrast', 'V-DB'))
            preds_v.extend(out_v.argmax(1).cpu().numpy())
            preds_db.extend(out_db.argmax(1).cpu().numpy())
            preds_fused.extend(out.argmax(1).cpu().numpy())
            targets.extend(target.numpy())
    return (np.array(preds_v), np.array(preds_db),
            np.array(preds_fused), np.array(targets))


def load_model(checkpoint_path, args):
    """Load model weights and optional RecorderMeter from a checkpoint."""
    model = DECNet(
        s_former_depth=args.s_former_depth,
        t_former_depth=args.t_former_depth,
        nf=args.nf,
        Incep_depth=args.Incep_depth,
        db_in_channels=args.db_in_channels,
        num_classes=args.num_classes,
    )
    model = torch.nn.DataParallel(model).cuda()
    ckpt     = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(ckpt['state_dict'])
    model.contrast = ckpt.get('contrast', args.contrast)
    args._db_normalization = ckpt.get('db_normalization')
    epoch    = ckpt.get('epoch', '?')
    best_acc = ckpt.get('best_acc', '?')
    if isinstance(best_acc, torch.Tensor):
        best_acc = f'{best_acc.item():.3f}'
    recorder = ckpt.get('recorder', None)
    return model, epoch, best_acc, recorder


def build_loader(txt_path, args):
    if args.dataset == 'AIDE':
        from dataloader.dataset_AIDE import DECNet_test_data_loader as _fn
    else:
        from dataloader.dataset_PPB import DECNet_test_data_loader as _fn
    test_data = _fn(
        data_set=1,
        txt_path=txt_path,
        contrast=args.contrast,
        t=args.t,
        db_pkl_path=args.db_pkl_path,
    )
    if getattr(args, '_db_normalization', None) is not None:
        from dataloader.normalization import apply_normalization
        apply_normalization(test_data, args._db_normalization)
    return torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(preds, targets):
    """
    Four metrics from the paper (§IV.B), all returned in percent:
      acc7        — overall top-1 accuracy          (Acc-7)
      f1_macro    — macro-averaged F1                (F1-7)
      acc_bal     — balanced accuracy = macro recall  (Acc)
      f1_weighted — weighted-averaged F1              (extra; paper's "F1" = per-class F1 in classification_report)
    """
    acc7        = 100.0 * (preds == targets).mean()
    f1_macro    = 100.0 * f1_score(targets, preds, average='macro',    zero_division=0)
    acc_bal     = 100.0 * balanced_accuracy_score(targets, preds)
    f1_weighted = 100.0 * f1_score(targets, preds, average='weighted', zero_division=0)
    return acc7, f1_macro, acc_bal, f1_weighted


def per_class_accuracy(preds, targets):
    """Per-class top-1 accuracy (%) for each emotion class."""
    accs = []
    for c in range(NUM_CLASSES):
        mask = targets == c
        accs.append(100.0 * (preds[mask] == targets[mask]).mean()
                    if mask.sum() > 0 else float('nan'))
    return accs


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _draw_confusion_matrix(ax, preds, targets, title='Confusion Matrix (row-normalised)'):
    """Draw a row-normalised confusion matrix onto ax."""
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)), normalize='true')
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_fs = max(6, 10 - NUM_CLASSES // 3)
    ax.set(
        xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
        xticklabels=EMOTIONS, yticklabels=EMOTIONS,
        xlabel='Predicted', ylabel='True',
        title=title,
    )
    ax.tick_params(axis='x', labelsize=tick_fs, rotation=45)
    ax.tick_params(axis='y', labelsize=tick_fs)
    thresh = 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f'{cm[i, j]:.2f}',
                    ha='center', va='center', fontsize=max(6, tick_fs - 1),
                    color='white' if cm[i, j] > thresh else 'black')


def generate_combined_figure(recorder, preds_v, preds_db, preds_fused,
                              targets, save_path, suptitle=''):
    """
    Save a combined PNG:
      Top row (when training curves exist in the checkpoint):
        Left:  training / validation accuracy over epochs
        Right: training / validation loss over epochs
      Bottom row (always):
        Three row-normalised confusion matrices: Visual | DB | Fused
        Predictions are pooled across all evaluated folds.
    """
    has_curves = recorder is not None and recorder.current_epoch > 0

    if has_curves:
        fig   = plt.figure(figsize=(18, 11))
        outer = gridspec.GridSpec(2, 1, figure=fig,
                                  height_ratios=[1.0, 1.5], hspace=0.42)
        top   = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.30)
        bot   = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.34)
        ax_acc  = fig.add_subplot(top[0])
        ax_loss = fig.add_subplot(top[1])

        n          = recorder.current_epoch
        epochs     = np.arange(1, n + 1)
        train_acc  = recorder.epoch_accuracy[:n, 0]
        val_acc    = recorder.epoch_accuracy[:n, 1]
        train_loss = recorder.epoch_losses[:n, 0] / 50.0   # undo ×50 from train.py
        val_loss   = recorder.epoch_losses[:n, 1] / 50.0

        ax_acc.plot(epochs, train_acc, color='#2196F3', lw=1.5, label='Train')
        ax_acc.plot(epochs, val_acc,   color='#FF9800', lw=1.5, label='Val', linestyle='--')
        best_epoch = int(val_acc.argmax()) + 1
        ax_acc.axvline(best_epoch, color='gray', lw=0.8, linestyle=':',
                       label=f'Best val (ep {best_epoch})')
        ax_acc.set(xlabel='Epoch', ylabel='Balanced Accuracy (Macro, %)',
                   title='Training / Validation Balanced Accuracy')
        ax_acc.set_xlim(1, n)
        ax_acc.set_ylim(0, 100)
        ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8, prune='both'))
        ax_acc.legend(fontsize=9)
        ax_acc.grid(True, alpha=0.3)

        ax_loss.plot(epochs, train_loss, color='#2196F3', lw=1.5, label='Train')
        ax_loss.plot(epochs, val_loss,   color='#FF9800', lw=1.5, label='Val', linestyle='--')
        ax_loss.axvline(best_epoch, color='gray', lw=0.8, linestyle=':',
                        label=f'Best val (ep {best_epoch})')
        ax_loss.set(xlabel='Epoch', ylabel='Loss', title='Training / Validation Loss')
        ax_loss.set_xlim(1, n)
        ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8, prune='both'))
        ax_loss.legend(fontsize=9)
        ax_loss.grid(True, alpha=0.3)

    else:
        fig   = plt.figure(figsize=(18, 6))
        outer = gridspec.GridSpec(1, 1, figure=fig)
        bot   = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.34)

    ax_cmv  = fig.add_subplot(bot[0])
    ax_cmdb = fig.add_subplot(bot[1])
    ax_cmf  = fig.add_subplot(bot[2])

    _draw_confusion_matrix(ax_cmv,  preds_v,     targets,
                           title='Visual branch\n(row-normalised, all folds pooled)')
    _draw_confusion_matrix(ax_cmdb, preds_db,    targets,
                           title='DB branch\n(row-normalised, all folds pooled)')
    _draw_confusion_matrix(ax_cmf,  preds_fused, targets,
                           title='Fused output\n(row-normalised, all folds pooled)')

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.01)

    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

# Column layout in per_fold_metrics tuples (12 values):
#   Fused:  0=acc7  1=f1_macro  2=acc_bal  3=f1_wtd
#   Visual: 4=acc7  5=f1_macro  6=acc_bal  7=f1_wtd
#   DB:     8=acc7  9=f1_macro 10=acc_bal 11=f1_wtd
_BRANCH_COLS = [
    ('**Fused**', 0,  1,  2,  3),
    ('Visual',    4,  5,  6,  7),
    ('DB',        8,  9, 10, 11),
]


def _average_recorders(recorders):
    """
    Return a single RecorderMeter whose epoch-wise values are the mean of
    all valid (non-None, non-empty) recorders.  Each epoch is averaged over
    only the folds that have data for that epoch, so a fold that stopped early
    (e.g. best model saved at epoch 3) does not truncate the whole curve.
    """
    valid = [r for r in recorders if r is not None and r.current_epoch > 0]
    if not valid:
        return None
    n   = max(r.current_epoch for r in valid)
    avg = RecorderMeter(valid[0].total_epoch)
    avg.current_epoch = n
    for t in range(n):
        contributing_acc  = [r.epoch_accuracy[t] for r in valid if r.current_epoch > t]
        contributing_loss = [r.epoch_losses[t]   for r in valid if r.current_epoch > t]
        avg.epoch_accuracy[t] = np.mean(contributing_acc, axis=0)
        avg.epoch_losses[t]   = np.mean(contributing_loss, axis=0)
    return avg


def _per_class_table(preds, targets, emotion_names):
    """Per-class Precision / Recall / F1 / Support table for log output."""
    prec, rec, f1, sup = precision_recall_fscore_support(
        targets, preds,
        labels=list(range(len(emotion_names))),
        zero_division=0,
    )
    bal_acc = balanced_accuracy_score(targets, preds)
    lines = [
        '| Class | Precision | Recall | F1-score | Support |',
        '|-------|-----------|--------|----------|---------|',
    ]
    for i, emo in enumerate(emotion_names):
        lines.append(
            f'| {emo} '
            f'| {prec[i]*100:.2f}% '
            f'| {rec[i]*100:.2f}% '
            f'| {f1[i]*100:.2f}% '
            f'| {int(sup[i])} |'
        )
    lines += [
        '| | | | | |',
        f'| **Macro avg** | {prec.mean()*100:.2f}% | {rec.mean()*100:.2f}% '
        f'| {f1.mean()*100:.2f}% | {int(sup.sum())} |',
        f'| **Balanced Acc** | — | **{bal_acc*100:.2f}%** | — | — |',
    ]
    return '\n'.join(lines)


def _ablation_table(preds_v, preds_db, preds_fused, targets, emotion_names):
    """
    Build the modality-ablation table from the paper (Table II style):

      | Setting | Emo1 Acc | Emo1 F1 | Emo2 Acc | Emo2 F1 | … | Acc-7 | F1-7 |
      |---------|----------|---------|----------|---------|---|-------|------|
      | FV      |          |         |          |         |   |       |      |
      | DB      |          |         |          |         |   |       |      |
      | FV+DB   |          |         |          |         |   |       |      |

    FV = Visual branch (facial videos)
    DB = DB branch (driving behaviour)
    FV+DB = Fused output

    Per-class Acc = per-class recall (fraction of that class correctly predicted).
    Acc-7 = balanced accuracy (macro-averaged recall).
    F1-7  = macro-averaged F1.
    """
    labels = list(range(len(emotion_names)))

    def _row(preds):
        _, rec, f1, _ = precision_recall_fscore_support(
            targets, preds, labels=labels, zero_division=0)
        acc7 = balanced_accuracy_score(targets, preds)
        f1_7 = f1_score(targets, preds, labels=labels, average='macro', zero_division=0)
        return rec, f1, acc7, f1_7

    branches = [
        ('FV',    _row(preds_v)),
        ('DB',    _row(preds_db)),
        ('FV+DB', _row(preds_fused)),
    ]

    # Header: one Acc+F1 pair per emotion, then Acc-7 and F1-7
    header_top = '| Setting |' + ''.join(
        f' {e} Acc | {e} F1 |' for e in emotion_names) + ' Acc-7 | F1-7 |'
    sep = '|---------|' + '---------|---------|' * len(emotion_names) + '-------|------|'

    rows = [header_top, sep]
    for label, (rec, f1, acc7, f17) in branches:
        cells = ''.join(
            f' {rec[i]*100:.2f} | {f1[i]*100:.2f} |' for i in range(len(emotion_names)))
        rows.append(f'| **{label}** |{cells} {acc7*100:.2f} | {f17*100:.2f} |')

    return '\n'.join(rows)


# ---------------------------------------------------------------------------
# Label grouping helpers
# ---------------------------------------------------------------------------

_DATASET_JSON_KEY = {'PPB': 'PPB-Emo', 'AIDE': 'AIDE'}


def build_group_mapping(json_path, grouping_name, dataset_type,
                         emotion_names, test_txt_path=None):
    """
    Derive a class_index → group_index mapping from label_groups.json.

    For PPB-Emo the JSON entries are "{participant}-{category}" strings
    (e.g. "P02-AD").  We extract the category suffix and map it to the
    EMOTIONS list, so the result is purely label-based and independent of
    which specific participant-clips appear in the test annotation.

    For AIDE the JSON entries are clip_ids (e.g. "0002").  We cross-reference
    with the test annotation file (which has {clip_id: class_label} pairs) to
    derive the class→group mapping.  test_txt_path is required for AIDE.

    Returns
    -------
    group_names : list[str]
        Ordered group names (e.g. ["Negative", "Neutral", "Positive"]).
    class_to_group : dict[int, int]
        Maps original class index → group index.  Missing entries mean that
        class has no group assignment in the JSON.
    """
    with open(json_path) as f:
        all_data = json.load(f)

    dataset_key = _DATASET_JSON_KEY.get(dataset_type, dataset_type)
    if dataset_key not in all_data:
        raise KeyError(
            f'Dataset key "{dataset_key}" not found in {json_path}. '
            f'Available: {list(all_data.keys())}')
    if grouping_name not in all_data[dataset_key]:
        raise KeyError(
            f'Grouping "{grouping_name}" not found under "{dataset_key}". '
            f'Available: {list(all_data[dataset_key].keys())}')

    groups_def = all_data[dataset_key][grouping_name]  # {group_name: [ids]}
    group_names = list(groups_def.keys())
    class_to_group = {}

    if dataset_type == 'PPB':
        # Entries like "P02-AD" → extract category "AD"
        name_to_group = {}
        for gi, (_, members) in enumerate(groups_def.items()):
            for m in members:
                cat = m.split('-', 1)[1]   # "P02-AD" → "AD"
                name_to_group[cat] = gi
        for ci, emo in enumerate(emotion_names):
            if emo in name_to_group:
                class_to_group[ci] = name_to_group[emo]

    else:  # AIDE — cross-reference clip_id → class via annotation file
        if test_txt_path is None:
            raise ValueError(
                'test_txt_path is required for AIDE label grouping.')
        clip_to_group = {}
        for gi, (_, members) in enumerate(groups_def.items()):
            for m in members:
                clip_to_group[str(m)] = gi

        with open(test_txt_path) as fh:
            for line in fh:
                fields = line.strip().split()
                if len(fields) < 4:
                    continue
                class_label = int(fields[2])
                clip_id = fields[3]
                if clip_id in clip_to_group and class_label not in class_to_group:
                    class_to_group[class_label] = clip_to_group[clip_id]

    return group_names, class_to_group


def apply_grouping(preds, class_to_group):
    """Remap class-index predictions using class_to_group; returns -1 for unknowns."""
    return np.array([class_to_group.get(int(p), -1) for p in preds])


def _grouped_ablation_table(preds_v, preds_db, preds_fused, targets,
                             group_names, class_to_group):
    """
    Build a per-group balanced-accuracy + F1 table after remapping original
    class predictions to the coarser group labels.

    Only samples whose true label maps to a known group are included.
    """
    g_targets = apply_grouping(targets, class_to_group)
    mask = g_targets >= 0
    if mask.sum() == 0:
        return '_No samples matched any group — check label_groups.json keys._'

    g_tgt = g_targets[mask]
    n_groups = len(group_names)
    labels = list(range(n_groups))

    def _row(preds):
        gp = apply_grouping(preds, class_to_group)[mask]
        # replace any -1 pred with 0 (predicted unknown → assigned to first group for scoring)
        gp = np.where(gp < 0, 0, gp)
        _, rec, f1, _ = precision_recall_fscore_support(
            g_tgt, gp, labels=labels, zero_division=0)
        acc_bal = balanced_accuracy_score(g_tgt, gp)
        f1_mac  = f1_score(g_tgt, gp, labels=labels, average='macro', zero_division=0)
        return rec, f1, acc_bal, f1_mac

    branches = [
        ('FV',    _row(preds_v)),
        ('DB',    _row(preds_db)),
        ('FV+DB', _row(preds_fused)),
    ]

    header = ('| Setting |'
              + ''.join(f' {g} Acc | {g} F1 |' for g in group_names)
              + ' Acc-bal | F1 |')
    sep = ('|---------|'
           + '---------|---------|' * n_groups
           + '---------|-----|')

    rows = [header, sep]
    for label, (rec, f1, acc_bal, f1_mac) in branches:
        cells = ''.join(
            f' {rec[i]*100:.2f} | {f1[i]*100:.2f} |' for i in range(n_groups))
        rows.append(f'| **{label}** |{cells} {acc_bal*100:.2f} | {f1_mac*100:.2f} |')

    n_used = int(mask.sum())
    n_total = len(targets)
    rows.append(f'\n_{n_used} of {n_total} test samples matched a group._')
    return '\n'.join(rows)


def generate_markdown_report(args, fold_pairs, per_fold_metrics, recorders,
                              all_pv, all_pdb, all_pf, all_t, figure_path, report_path,
                              train_summary_override=None,
                              group_sections=None):
    n_folds = len(fold_pairs)
    multi   = n_folds > 1

    ablation = _ablation_table(all_pv, all_pdb, all_pf, all_t, EMOTIONS)

    if train_summary_override is not None:
        train_summary = train_summary_override
    else:
        recorder = recorders[0] if recorders else None
        if recorder is not None and recorder.current_epoch > 0:
            n        = recorder.current_epoch
            best_val = recorder.epoch_accuracy[:n, 1].max()
            best_ep  = int(recorder.epoch_accuracy[:n, 1].argmax()) + 1
            train_summary = (
                f'Best validation accuracy during training: **{best_val:.2f}%** '
                f'(epoch {best_ep} / {n})'
            )
            if multi:
                train_summary += '  \n_Training curves averaged across inner CV folds._'
        else:
            train_summary = '_Training curves not available (recorder not found in checkpoint)._'

    fig_rel = os.path.relpath(figure_path, os.path.dirname(report_path))

    lines = [
        '# DECNet Evaluation Report',
        '',
        f'**Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  ',
        f'**Dataset**: {args.dataset}  ',
        f'**Model**: S={args.s_former_depth}, T={args.t_former_depth}, '
        f'I={args.Incep_depth}, nf={args.nf}  ',
        f'**Contrast**: {args.contrast}  ',
        f'**Clip duration**: {args.t} s  ',
        f'**Inner CV folds**: {n_folds}  ',
        '',
        '---',
        '',
        '## Training Summary',
        '',
        train_summary,
        '',
        '---',
        '',
        '## Modality Ablation (all test samples pooled)',
        '',
        '> FV = facial video branch  ·  DB = driving-behaviour branch  ·  FV+DB = fused  ',
        '> Per-class Acc = per-class recall (fraction of that class correctly predicted).  ',
        '> Acc-7 = balanced accuracy (macro-averaged recall).  F1-7 = macro-averaged F1.',
        '',
        ablation,
        '',
        '---',
        '',
        '## Visualisation',
        '',
        f'![Training curves and confusion matrices]({fig_rel})',
        '',
        '_Top row: training/validation balanced accuracy and loss averaged across inner CV folds.  '
        'Bottom row: row-normalised confusion matrices for the Visual (FV), DB, and Fused (FV+DB) '
        'branches (all test samples pooled).  '
        'Dotted vertical line marks the best validation epoch._',
        '',
        '---',
        '',
        '_Generated by `test.py` — DECNet evaluation._',
    ]

    if group_sections:
        lines = lines[:-1]   # remove trailing generated-by line
        for grouping_name, group_table in group_sections:
            lines += [
                '',
                '---',
                '',
                f'## Grouped Label Analysis: {grouping_name}',
                '',
                ('> Original class predictions are remapped to coarser groups defined '
                 f'in `label_groups.json`.  '
                 'Acc-bal = balanced accuracy over group labels.  '
                 'F1 = macro F1 over group labels.'),
                '',
                group_table,
            ]
        lines += ['', '_Generated by `test.py` — DECNet evaluation._']

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def log(msg, f):
    print(msg)
    f.write(msg + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DECNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cv = parser.add_argument_group('CV mode (primary — fixed test set + inner CV folds)')
    cv.add_argument('--cv_last_checkpoints', nargs='+', default=None, metavar='CKPT',
                    help='Last (epoch-final) checkpoint per inner fold, used only for '
                         'training curve display. When provided, replaces best checkpoints '
                         'as the recorder source so curves span the full training run.')
    cv.add_argument('--cv_checkpoints', nargs='+', default=None, metavar='CKPT',
                    help='one best-checkpoint per inner CV fold; best is auto-selected '
                         'by val balanced accuracy and evaluated on --test_txt_path')
    cv.add_argument('--test_txt_path', default=None, type=str,
                    help='held-out test annotation .txt file')

    sg = parser.add_argument_group('Single-fold mode (legacy)')
    sg.add_argument('--checkpoint', default=None, type=str,
                    help='path to a single checkpoint')

    kf = parser.add_argument_group('Rotating-test-set mode (legacy, paper protocol)')
    kf.add_argument('--checkpoints', nargs='+', default=None, metavar='CKPT',
                    help='one checkpoint per rotating test fold')
    kf.add_argument('--annotation_dir', default='annotation', type=str,
                    help='directory containing annotation .txt files')
    kf.add_argument('--n_folds', default=5, type=int,
                    help='number of folds')

    sh = parser.add_argument_group('Shared options')
    sh.add_argument('--db_pkl_path', default=None, type=str,
                    help='path to DB feature pickle (None → original hardcoded path)')
    sh.add_argument('-t', default=3, type=int,
                    help='clip duration in seconds; must match pickle and annotation files')
    sh.add_argument('--contrast', default='V-DB', choices=['V-DB', 'V', 'DB'],
                    help='V-DB: both streams, V: visual only, DB: DB only')
    sh.add_argument('-b', '--batch-size', default=32, type=int)
    sh.add_argument('-j', '--workers',    default=8,  type=int)
    sh.add_argument('--s_former_depth', default=1,  type=int,
                    help='ViT layers in spatial transformer (paper optimal: 1)')
    sh.add_argument('--t_former_depth', default=3,  type=int,
                    help='transformer layers in temporal transformer (paper optimal: 3)')
    sh.add_argument('--nf',          default=32, type=int,
                    help='InceptionTime base feature width')
    sh.add_argument('--Incep_depth', default=6,  type=int,
                    help='number of Inception modules (paper optimal: 6)')
    sh.add_argument('--output_dir', default='log', type=str,
                    help='directory for all output files')
    sh.add_argument('--dataset', default='PPB', choices=['PPB', 'AIDE'],
                    help='Dataset loader: PPB (default) or AIDE')
    sh.add_argument('--db_in_channels', default=8, type=int,
                    help='DB input channels: 8 for PPB, 195 for AIDE')
    sh.add_argument('--num_classes', default=7, type=int,
                    help='Emotion classes: 7 for PPB, 5 for AIDE')
    sh.add_argument('--emotions', default=None, type=str,
                    help='Comma-separated emotion class names '
                         '(e.g. "Anger,Anxiety,Happiness,Peace,Weariness" for AIDE)')
    sh.add_argument('--label_groups_json', default=None, type=str,
                    help='Path to label_groups.json; all groupings defined for the '
                         'dataset (valence, driving_safety, …) are run automatically '
                         '(e.g. /data/gianluca/scripts/outputs/preprocessing/label_groups.json)')

    args = parser.parse_args()

    global EMOTIONS, NUM_CLASSES
    if args.emotions:
        EMOTIONS    = [e.strip() for e in args.emotions.split(',')]
        NUM_CLASSES = len(EMOTIONS)
    else:
        NUM_CLASSES = args.num_classes

    cudnn.benchmark    = False
    cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # ----------------------------------------------------------------
    # CV mode: inner CV checkpoints + fixed test set
    # ----------------------------------------------------------------
    if args.cv_checkpoints is not None and args.test_txt_path:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path    = os.path.join(args.output_dir, 'eval-log.txt')
        figure_path = os.path.join(args.output_dir, 'combined-figure.png')
        report_path = os.path.join(args.output_dir, 'eval-report.md')

        HDR = (f'  {"Branch":<8} {"Acc-7":>8}  {"F1-macro":>10}'
               f'  {"Acc-bal":>9}  {"F1-wtd":>8}')

        with open(log_path, 'w') as f:
            log(f'DECNet evaluation — CV mode  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', f)
            log(f'dataset={args.dataset}  contrast={args.contrast}  t={args.t}s  '
                f's={args.s_former_depth}  T={args.t_former_depth}  '
                f'I={args.Incep_depth}  nf={args.nf}', f)
            log(f'inner CV checkpoints: {len(args.cv_checkpoints)}', f)
            log('', f)

            # Phase 1 — read best_acc and recorder from every CV checkpoint (CPU only).
            # Recorders come from the last checkpoint (full training history) when
            # --cv_last_checkpoints is provided; otherwise fall back to best checkpoints.
            cv_best_accs = []
            cv_recorders = []
            recorder_ckpts = (args.cv_last_checkpoints
                              if args.cv_last_checkpoints is not None
                              else args.cv_checkpoints)
            for ckpt_path, rec_path in zip(args.cv_checkpoints, recorder_ckpts):
                ckpt_data = torch.load(ckpt_path, map_location='cpu')
                ba = ckpt_data.get('best_acc', 0.0)
                if hasattr(ba, 'item'):
                    ba = ba.item()
                cv_best_accs.append(float(ba))
                if rec_path != ckpt_path:
                    rec_data = torch.load(rec_path, map_location='cpu')
                else:
                    rec_data = ckpt_data
                cv_recorders.append(rec_data.get('recorder', None))
                log(f'  {os.path.basename(ckpt_path):<50} val_bal_acc={ba:.3f}%', f)

            # Phase 2 — select the inner fold whose checkpoint has the highest val balanced acc
            best_fold_idx = int(np.argmax(cv_best_accs))
            log('', f)
            log(f'Best inner fold: {best_fold_idx}  '
                f'(val balanced acc = {cv_best_accs[best_fold_idx]:.3f}%)', f)

            # Phase 3 — evaluate best model on held-out test set
            model, best_epoch, _, _ = load_model(args.cv_checkpoints[best_fold_idx], args)
            loader = build_loader(args.test_txt_path, args)
            pv, pdb, pfused, tgt = evaluate_fold(model, loader)
            del model
            torch.cuda.empty_cache()

            acc_f,  f1_f,  acc_bal_f,  f1_w_f  = compute_metrics(pfused, tgt)
            acc_v,  f1_v,  acc_bal_v,  f1_w_v  = compute_metrics(pv,     tgt)
            acc_db, f1_db, acc_bal_db, f1_w_db = compute_metrics(pdb,    tgt)

            log('', f)
            log('=== TEST SET RESULTS (best inner fold model) ===', f)
            log(HDR, f)
            log(f'  {"Fused":<8} {acc_f:>7.2f}%  {f1_f:>9.2f}%'
                f'  {acc_bal_f:>8.2f}%  {f1_w_f:>7.2f}%', f)
            log(f'  {"Visual":<8} {acc_v:>7.2f}%  {f1_v:>9.2f}%'
                f'  {acc_bal_v:>8.2f}%  {f1_w_v:>7.2f}%', f)
            log(f'  {"DB":<8} {acc_db:>7.2f}%  {f1_db:>9.2f}%'
                f'  {acc_bal_db:>8.2f}%  {f1_w_db:>7.2f}%', f)
            log('', f)
            for bname, preds_all in [('Fused', pfused), ('Visual', pv), ('DB', pdb)]:
                log(f'--- Per-class statistics ({bname}) ---', f)
                log(_per_class_table(preds_all, tgt, EMOTIONS), f)
                log('', f)

            # Phase 4 — average training curves across all inner CV folds
            avg_recorder = _average_recorders(cv_recorders)

        # Generate figure (averaged curves + test confusion matrices)
        model_cfg = (f'S={args.s_former_depth}, T={args.t_former_depth}, '
                     f'I={args.Incep_depth}, nf={args.nf}, {args.contrast}')
        generate_combined_figure(
            avg_recorder, pv, pdb, pfused, tgt, figure_path,
            suptitle=(f'DECNet — {args.dataset} — {model_cfg}  |  '
                      f'best inner fold {best_fold_idx} on held-out test set'),
        )

        cv_summary = (
            f'Training curves averaged across {len(args.cv_checkpoints)} inner CV folds.  \n'
            f'**Best inner fold**: {best_fold_idx} '
            f'(val balanced acc = {cv_best_accs[best_fold_idx]:.2f}%)  \n'
            f'Confusion matrices and per-class statistics are from that model '
            f'evaluated on the held-out test set.'
        )

        group_sections = []
        if args.label_groups_json:
            with open(args.label_groups_json) as _jf:
                _all_groups = json.load(_jf)
            _key = _DATASET_JSON_KEY.get(args.dataset, args.dataset)
            for gname in _all_groups.get(_key, {}).keys():
                try:
                    gnames, c2g = build_group_mapping(
                        args.label_groups_json, gname,
                        args.dataset, EMOTIONS, args.test_txt_path)
                    gtable = _grouped_ablation_table(pv, pdb, pfused, tgt, gnames, c2g)
                    group_sections.append((gname, gtable))
                    with open(log_path, 'a') as f:
                        log(f'\n--- Grouped label analysis: {gname} '
                            f'({", ".join(gnames)}) ---', f)
                        log(gtable, f)
                except Exception as e:
                    print(f'[WARN] grouping "{gname}" skipped: {e}')

        generate_markdown_report(
            args,
            fold_pairs=[(args.cv_checkpoints[best_fold_idx], args.test_txt_path)],
            per_fold_metrics=[(acc_f, f1_f, acc_bal_f, f1_w_f,
                               acc_v, f1_v, acc_bal_v, f1_w_v,
                               acc_db, f1_db, acc_bal_db, f1_w_db)],
            recorders=[avg_recorder],
            all_pv=pv, all_pdb=pdb, all_pf=pfused, all_t=tgt,
            figure_path=figure_path, report_path=report_path,
            train_summary_override=cv_summary,
            group_sections=group_sections,
        )

        with open(log_path, 'a') as f:
            log(f'\nFigure  → {figure_path}', f)
            log(f'Report  → {report_path}', f)
            log(f'Log     → {log_path}', f)
        return

    # ----------------------------------------------------------------
    # Legacy modes: build (checkpoint, annotation) pairs
    # ----------------------------------------------------------------
    if args.checkpoints is not None:
        if len(args.checkpoints) != args.n_folds:
            parser.error(f'--checkpoints: expected {args.n_folds} paths, '
                         f'got {len(args.checkpoints)}')
        fold_pairs = [
            (ckpt, os.path.join(
                args.annotation_dir,
                f'PPB_CIR_V_DB_rgb_112_{args.t}s_test_fold_{i}.txt'))
            for i, ckpt in enumerate(args.checkpoints)
        ]
    elif args.checkpoint and args.test_txt_path:
        fold_pairs = [(args.checkpoint, args.test_txt_path)]
    else:
        parser.error('Provide --cv_checkpoints + --test_txt_path (CV mode), '
                     '--checkpoint + --test_txt_path (single), '
                     'or --checkpoints (rotating test folds).')

    os.makedirs(args.output_dir, exist_ok=True)
    log_path    = os.path.join(args.output_dir, 'eval-log.txt')
    figure_path = os.path.join(args.output_dir, 'combined-figure.png')
    report_path = os.path.join(args.output_dir, 'eval-report.md')

    fold_preds_v, fold_preds_db, fold_preds_fused, fold_targets = [], [], [], []
    per_fold_metrics = []
    recorders        = []

    HDR = (f'  {"Branch":<8} {"Acc-7":>8}  {"F1-macro":>10}'
           f'  {"Acc-bal":>9}  {"F1-wtd":>8}')

    with open(log_path, 'w') as f:
        log(f'DECNet evaluation  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', f)
        log(f'dataset={args.dataset}  contrast={args.contrast}  t={args.t}s  '
            f's={args.s_former_depth}  T={args.t_former_depth}  '
            f'I={args.Incep_depth}  nf={args.nf}', f)
        log('Metrics: Acc-7 (overall acc), F1-macro (macro F1), '
            'Acc-bal (balanced acc), F1-wtd (weighted F1)', f)
        log('', f)

        for fold_idx, (ckpt_path, txt_path) in enumerate(fold_pairs):
            log(f'=== Fold {fold_idx} ===', f)
            log(f'  checkpoint : {ckpt_path}', f)
            log(f'  test set   : {txt_path}', f)

            model, ckpt_epoch, ckpt_best, recorder = load_model(ckpt_path, args)
            recorders.append(recorder)
            log(f'  saved at epoch {ckpt_epoch},  best_acc={ckpt_best}', f)
            if recorder is None:
                log('  recorder: not in checkpoint — training curves unavailable', f)

            loader = build_loader(txt_path, args)
            pv, pdb, pfused, tgt = evaluate_fold(model, loader)

            fold_preds_v.append(pv)
            fold_preds_db.append(pdb)
            fold_preds_fused.append(pfused)
            fold_targets.append(tgt)

            acc_f,  f1_f,  acc_bal_f,  f1_w_f  = compute_metrics(pfused, tgt)
            acc_v,  f1_v,  acc_bal_v,  f1_w_v  = compute_metrics(pv,     tgt)
            acc_db, f1_db, acc_bal_db, f1_w_db = compute_metrics(pdb,    tgt)

            per_fold_metrics.append((
                acc_f,  f1_f,  acc_bal_f,  f1_w_f,
                acc_v,  f1_v,  acc_bal_v,  f1_w_v,
                acc_db, f1_db, acc_bal_db, f1_w_db,
            ))

            log(HDR, f)
            log(f'  {"Fused":<8} {acc_f:>7.2f}%  {f1_f:>9.2f}%'
                f'  {acc_bal_f:>8.2f}%  {f1_w_f:>7.2f}%', f)
            log(f'  {"Visual":<8} {acc_v:>7.2f}%  {f1_v:>9.2f}%'
                f'  {acc_bal_v:>8.2f}%  {f1_w_v:>7.2f}%', f)
            log(f'  {"DB":<8} {acc_db:>7.2f}%  {f1_db:>9.2f}%'
                f'  {acc_bal_db:>8.2f}%  {f1_w_db:>7.2f}%', f)
            log('', f)

            del model
            torch.cuda.empty_cache()

        # ----------------------------------------------------------------
        # Aggregate across folds
        # ----------------------------------------------------------------
        all_pv  = np.concatenate(fold_preds_v)
        all_pdb = np.concatenate(fold_preds_db)
        all_pf  = np.concatenate(fold_preds_fused)
        all_t   = np.concatenate(fold_targets)
        m       = np.array(per_fold_metrics)   # (n_folds, 12)

        log('=' * 72, f)
        log('SUMMARY', f)
        log('=' * 72, f)

        if len(fold_pairs) > 1:
            log(f'  {"Branch":<8} {"Acc-7":>22}  {"F1-macro":>22}'
                f'  {"Acc-bal":>22}  {"F1-wtd":>22}', f)
            for name, ca, cf, cb, cw in _BRANCH_COLS:
                name_plain = name.replace('**', '')
                a = f'{m[:, ca].mean():.2f} ± {m[:, ca].std():.2f}%'
                b = f'{m[:, cf].mean():.2f} ± {m[:, cf].std():.2f}%'
                c = f'{m[:, cb].mean():.2f} ± {m[:, cb].std():.2f}%'
                d = f'{m[:, cw].mean():.2f} ± {m[:, cw].std():.2f}%'
                log(f'  {name_plain:<8} {a:>22}  {b:>22}  {c:>22}  {d:>22}', f)
        else:
            row = per_fold_metrics[0]
            log(HDR, f)
            for name, ca, cf, cb, cw in _BRANCH_COLS:
                name_plain = name.replace('**', '')
                log(f'  {name_plain:<8} {row[ca]:>7.2f}%  {row[cf]:>9.2f}%'
                    f'  {row[cb]:>8.2f}%  {row[cw]:>7.2f}%', f)

        log('', f)

        for branch_name, preds_all in [('Fused', all_pf),
                                        ('Visual', all_pv),
                                        ('DB', all_pdb)]:
            log(f'--- Per-class statistics ({branch_name}, all folds pooled) ---', f)
            log(_per_class_table(preds_all, all_t, EMOTIONS), f)

        log('Per-class accuracy (fused, all folds pooled):', f)
        for emo, acc in zip(EMOTIONS, per_class_accuracy(all_pf, all_t)):
            log(f'  {emo:4s} : {"N/A" if np.isnan(acc) else f"{acc:.2f}%"}', f)

        # ----------------------------------------------------------------
        # Outputs: combined figure + Markdown report
        # ----------------------------------------------------------------
        model_cfg = (f'S={args.s_former_depth}, T={args.t_former_depth}, '
                     f'I={args.Incep_depth}, nf={args.nf}, {args.contrast}')
        generate_combined_figure(
            recorders[0], all_pv, all_pdb, all_pf, all_t, figure_path,
            suptitle=f'DECNet — {args.dataset} — {model_cfg}',
        )

        group_sections = []
        if args.label_groups_json:
            ref_txt = fold_pairs[0][1] if fold_pairs else None
            with open(args.label_groups_json) as _jf:
                _all_groups = json.load(_jf)
            _key = _DATASET_JSON_KEY.get(args.dataset, args.dataset)
            for gname in _all_groups.get(_key, {}).keys():
                try:
                    gnames, c2g = build_group_mapping(
                        args.label_groups_json, gname,
                        args.dataset, EMOTIONS, ref_txt)
                    gtable = _grouped_ablation_table(
                        all_pv, all_pdb, all_pf, all_t, gnames, c2g)
                    group_sections.append((gname, gtable))
                    log(f'\n--- Grouped label analysis: {gname} '
                        f'({", ".join(gnames)}) ---', f)
                    log(gtable, f)
                except Exception as e:
                    print(f'[WARN] grouping "{gname}" skipped: {e}')

        generate_markdown_report(
            args, fold_pairs, per_fold_metrics, recorders,
            all_pv, all_pdb, all_pf, all_t, figure_path, report_path,
            group_sections=group_sections,
        )

        log(f'\nCombined figure  → {figure_path}', f)
        log(f'Markdown report  → {report_path}', f)
        log(f'Log              → {log_path}', f)


if __name__ == '__main__':
    main()
