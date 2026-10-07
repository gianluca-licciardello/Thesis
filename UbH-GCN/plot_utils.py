"""Shared plotting utilities for main.py and ensemble.py."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

EMOTION_LABELS = ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger']


def _render_confusion_matrix(
    cm: np.ndarray,
    out_path: str,
    labels: list[str],
    title: str,
    acc: float,
    macro_acc: float,
    f1_w: float,
    macro_f1: float,
    prec: np.ndarray,
    rec: np.ndarray,
    f1_pc: np.ndarray,
) -> None:
    n = len(labels)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums

    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[5, 1], hspace=0.45, figure=fig,
        left=0.12, right=0.92, top=0.88, bottom=0.08,
    )
    ax_cm  = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis('off')

    im = ax_cm.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label('Recall (row-normalised)', rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax_cm.set_xticks(range(n))
    ax_cm.set_yticks(range(n))
    ax_cm.set_xticklabels(labels, rotation=40, ha='right', fontsize=11)
    ax_cm.set_yticklabels(labels, fontsize=11)
    ax_cm.set_xlabel('Predicted', fontsize=12, labelpad=8)
    ax_cm.set_ylabel('True', fontsize=12, labelpad=8)

    for i in range(n):
        for j in range(n):
            pct = cm_norm[i, j]
            color = 'white' if pct > 0.55 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax_cm.text(
                j, i,
                f'{int(cm[i, j])}\n{pct * 100:.2f}%',
                ha='center', va='center',
                color=color, fontsize=10, fontweight=weight,
            )

    heading = f'Confusion Matrix  —  {title}' if title else 'Confusion Matrix'
    ax_cm.set_title(
        f'{heading}\n'
        f'Accuracy: {acc * 100:.2f}%    Macro Acc: {macro_acc * 100:.2f}%\n'
        f'Weighted F1: {f1_w * 100:.2f}%    Macro F1: {macro_f1 * 100:.2f}%',
        fontsize=11, pad=10,
    )

    col_labels = ['Class', 'Precision', 'Recall', 'F1-score', 'Support']
    table_data = [
        [
            labels[i],
            f'{prec[i] * 100:.2f}%',
            f'{rec[i] * 100:.2f}%',
            f'{f1_pc[i] * 100:.2f}%',
            str(int(row_sums[i, 0])),
        ]
        for i in range(n)
    ]
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#4472C4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(n):
        tbl[i + 1, 0].set_facecolor('#dce6f1')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_confusion_matrix(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    out_path: str,
    labels: list[str] = EMOTION_LABELS,
    title: str = '',
) -> tuple[float, float, np.ndarray]:
    """Save a confusion matrix figure from raw prediction arrays.

    The heatmap is row-normalised (each row = recall of that class).
    Each cell shows the raw count (bold) and the row percentage below it.
    Overall accuracy, macro accuracy, and weighted F1 appear in the title.
    A per-class metrics table (Precision / Recall / F1) is rendered below.

    Returns:
        (accuracy, weighted_f1, confusion_matrix_array)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(labels)

    cm    = confusion_matrix(y_true, y_pred, labels=range(n))
    acc   = accuracy_score(y_true, y_pred)
    f1_w  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec  = precision_score(y_true, y_pred, average=None, labels=range(n), zero_division=0)
    rec   = recall_score(y_true, y_pred, average=None, labels=range(n), zero_division=0)
    f1_pc = f1_score(y_true, y_pred, average=None, labels=range(n), zero_division=0)
    macro_acc = float(np.mean(rec))
    macro_f1  = float(np.mean(f1_pc))

    _render_confusion_matrix(cm, out_path, labels, title, acc, macro_acc, f1_w, macro_f1, prec, rec, f1_pc)
    return acc, f1_w, cm


def save_confusion_matrix_from_cm(
    cm: np.ndarray,
    out_path: str,
    labels: list[str] = EMOTION_LABELS,
    title: str = '',
) -> None:
    """Save a confusion matrix figure from a pre-computed integer CM array.

    All metrics (accuracy, macro accuracy, weighted F1, per-class P/R/F1)
    are derived from the CM, so no raw prediction arrays are needed.
    """
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1).clip(min=1)
    col_sums = cm.sum(axis=0).clip(min=1)
    total = cm.sum()

    rec   = np.diag(cm) / row_sums
    prec  = np.diag(cm) / col_sums
    f1_pc = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    acc       = float(np.trace(cm) / total) if total > 0 else 0.0
    macro_acc = float(np.mean(rec))
    f1_w      = float(np.sum(f1_pc * row_sums / total))
    macro_f1  = float(np.mean(f1_pc))

    _render_confusion_matrix(cm.astype(int), out_path, labels, title,
                              acc, macro_acc, f1_w, macro_f1, prec, rec, f1_pc)
