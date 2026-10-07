"""
report.py – per-model summary table and training curves.

Reads log.txt and summary.md written by main.py; does not modify any other file.

Usage:
    python report.py --main-dir ./work_dir/aide_fast_sam3d/
    python report.py --main-dir ./work_dir/ppb_emo_fast_sam3d_mixed/

Outputs (written to --main-dir):
    single_models_summary_table.png
    training_curves.png
"""

import argparse
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_utils import save_confusion_matrix_from_cm

_LABELS = {
    5: ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger'],
    7: ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'],
}


# ── log parsing ──────────────────────────────────────────────────────────────

def _parse_log(log_path):
    """Return per-epoch metrics from the last complete training run in log.txt."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        lines = f.readlines()

    # Use only lines belonging to the last training run (restart-safe).
    last_start = 0
    for i, line in enumerate(lines):
        # Match epoch 1 exactly; a substring check also matches epochs 10-19
        # and incorrectly truncates a 90-epoch run to its final 72 epochs.
        if re.search(r'Training epoch: 1\b', line):
            last_start = i
    lines = lines[last_start:]

    train_loss, train_acc, eval_acc, eval_loss = [], [], [], []
    for line in lines:
        m = re.search(r'Mean training loss: (\d+(?:\.\d+)?|nan).*Mean training acc: (\d+(?:\.\d+)?)%', line)
        if m:
            val = float(m.group(1)) if m.group(1) != 'nan' else float('nan')
            train_loss.append(val)
            train_acc.append(float(m.group(2)))

        m = re.search(r'Mean eval acc: (\d+(?:\.\d+)?)%', line)
        if m:
            eval_acc.append(float(m.group(1)))

        m = re.search(r'Mean eval loss of \d+ batches: (\d+(?:\.\d+)?|nan)', line)
        if m:
            val = float(m.group(1)) if m.group(1) != 'nan' else float('nan')
            eval_loss.append(val)

    return {
        'train_loss': train_loss,
        'train_acc':  train_acc,
        'eval_acc':   eval_acc,
        'eval_loss':  eval_loss,
    }


def _parse_summary(summary_path):
    """Return scalar test metrics from summary.md."""
    if not os.path.exists(summary_path):
        return {}
    result = {}
    with open(summary_path) as f:
        for line in f:
            m = re.search(r'Best eval accuracy\s*\|\s*([\d.]+)%', line)
            if m:
                result['best_eval_acc'] = float(m.group(1))
            m = re.search(r'Test accuracy\s*\|\s*([\d.]+)%', line)
            if m:
                result['test_acc'] = float(m.group(1))
            m = re.search(r'Test weighted F1\s*\|\s*([\d.]+)%', line)
            if m:
                result['test_f1'] = float(m.group(1))
    return result


def _parse_class_csv(csv_path):
    """Parse epoch*_each_class_acc.csv; return (macro_acc_pct, macro_f1_pct).

    CSV layout written by main.py:
      row 0  : per-class recall (fractions 0-1)
      rows 1+: confusion matrix (integer counts)

    Macro accuracy = mean of per-class recalls.
    Macro F1      = unweighted mean of per-class F1 scores derived from the
                    confusion matrix (equal weight per class, suitable for
                    imbalanced datasets).
    Returns (None, None) if the file is missing or cannot be parsed.
    """
    if not os.path.exists(csv_path):
        return None, None
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, None
    try:
        class_acc = [float(v) for v in rows[0] if v.strip()]
        if not class_acc:
            return None, None
        macro_acc = float(np.mean(class_acc)) * 100

        macro_f1 = None
        if len(rows) > 1:
            cm = np.array([[float(v) for v in r if v.strip()] for r in rows[1:]])
            if cm.ndim == 2 and cm.shape[0] == cm.shape[1] == len(class_acc):
                f1s = []
                for i in range(len(class_acc)):
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    f1s.append(f1)
                macro_f1 = float(np.mean(f1s)) * 100

        return macro_acc, macro_f1
    except (ValueError, IndexError):
        return None, None


def _load_model(work_dir):
    log     = _parse_log(os.path.join(work_dir, 'log.txt'))
    summary = _parse_summary(os.path.join(work_dir, 'summary.md'))
    if log is not None and log['train_acc']:
        summary['best_train_acc'] = max(log['train_acc'])
        best_train_epoch = int(np.argmax(log['train_acc'])) + 1
        macro_acc_tr, macro_f1_tr = _parse_class_csv(
            os.path.join(work_dir, f'epoch{best_train_epoch}_train_each_class_acc.csv'))
        if macro_acc_tr is not None:
            summary['best_train_macro_acc'] = macro_acc_tr
        if macro_f1_tr is not None:
            summary['best_train_macro_f1'] = macro_f1_tr
    # summary.md may be overwritten by a subsequent --phase test run (which
    # resets best_acc to 0), so always derive best_eval_acc from the log.
    if log is not None and log['eval_acc']:
        summary['best_eval_acc'] = max(log['eval_acc'])
        # Best eval epoch (1-indexed) — used to locate the right CSV file.
        best_epoch = int(np.argmax(log['eval_acc'])) + 1
        macro_acc, macro_f1 = _parse_class_csv(
            os.path.join(work_dir, f'epoch{best_epoch}_eval_each_class_acc.csv'))
        if macro_acc is not None:
            summary['best_eval_macro_acc'] = macro_acc
        if macro_f1 is not None:
            summary['best_eval_macro_f1'] = macro_f1
    # Test macro accuracy / F1 from the per-class CSV written by --phase test.
    macro_acc_t, macro_f1_t = _parse_class_csv(
        os.path.join(work_dir, 'epoch1_test_each_class_acc.csv'))
    if macro_acc_t is not None:
        summary['test_macro_acc'] = macro_acc_t
    if macro_f1_t is not None:
        summary['test_macro_f1'] = macro_f1_t
    return log, summary


# ── table ────────────────────────────────────────────────────────────────────

def _fmt(v):
    return f'{v:.2f}%' if v is not None else 'N/A'


def save_summary_table(main_dir, model_keys, titles):
    col_labels = [
        'Model',
        'Best Train Acc', 'Best Train Macro',
        'Best Eval Acc', 'Best Eval Macro',
        'Test Acc', 'Test Macro Acc',
        'Test W.F1', 'Test Macro F1',
    ]
    rows = []
    for key, title in zip(model_keys, titles):
        work_dir = os.path.join(main_dir, key)
        _, summary = _load_model(work_dir)
        rows.append([
            title,
            _fmt(summary.get('best_train_acc')),
            _fmt(summary.get('best_train_macro_acc')),
            _fmt(summary.get('best_eval_acc')),
            _fmt(summary.get('best_eval_macro_acc')),
            _fmt(summary.get('test_acc')),
            _fmt(summary.get('test_macro_acc')),
            _fmt(summary.get('test_f1')),
            _fmt(summary.get('test_macro_f1')),
        ])

    # console
    print('\n=== Single-Model Results ===')
    header = '  '.join(f'{c:<18}' for c in col_labels)
    print(header)
    print('-' * len(header))
    for row in rows:
        print('  '.join(f'{v:<18}' for v in row))

    # PNG — wider figure to fit 9 columns
    fig, ax = plt.subplots(figsize=(23, 2.4))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#4472C4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(len(rows)):
        tbl[i + 1, 0].set_facecolor('#dce6f1')
    fig.suptitle('Single-Model Performance Summary', fontsize=12, fontweight='bold', y=0.98)
    out_path = os.path.join(main_dir, 'single_models_summary_table.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary table saved to {out_path}')


# ── training curves ───────────────────────────────────────────────────────────

def save_training_curves(main_dir, model_keys, titles):
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    fig.suptitle('Training Curves per Model', fontsize=14, fontweight='bold')

    for row, (key, title) in enumerate(zip(model_keys, titles)):
        ax_loss, ax_acc = axes[row, 0], axes[row, 1]
        work_dir = os.path.join(main_dir, key)
        log, _ = _load_model(work_dir)

        if log is None:
            for ax in (ax_loss, ax_acc):
                ax.text(0.5, 0.5, 'log.txt not found', ha='center', va='center',
                        transform=ax.transAxes, color='grey')
            ax_loss.set_title(f'{title}  –  Loss', fontsize=10)
            ax_acc.set_title(f'{title}  –  Accuracy', fontsize=10)
            continue

        epochs_tr = range(1, len(log['train_loss']) + 1)
        epochs_ev = range(1, len(log['eval_acc']) + 1)

        ax_loss.plot(epochs_tr, log['train_loss'], label='Train', color='steelblue')
        if log['eval_loss']:
            ax_loss.plot(epochs_ev, log['eval_loss'], label='Eval', color='darkorange')
        ax_loss.set_title(f'{title}  –  Loss', fontsize=10)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        ax_acc.plot(epochs_tr, log['train_acc'], label='Train', color='steelblue')
        if log['eval_acc']:
            ax_acc.plot(epochs_ev, log['eval_acc'], label='Eval', color='darkorange')
        ax_acc.set_title(f'{title}  –  Accuracy (%)', fontsize=10)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(main_dir, 'training_curves.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Training curves saved to {out_path}')


# ── per-model confusion matrices ─────────────────────────────────────────────

def _parse_cm_from_csv(csv_path):
    """Return the integer confusion matrix from epoch1_test_each_class_acc.csv, or None."""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None
    try:
        class_acc = [float(v) for v in rows[0] if v.strip()]
        n = len(class_acc)
        cm = np.array([[float(v) for v in r if v.strip()] for r in rows[1:]])
        if cm.ndim == 2 and cm.shape[0] == cm.shape[1] == n:
            return cm.astype(int)
    except (ValueError, IndexError):
        pass
    return None


def save_per_model_confusion_matrices(main_dir, model_keys, titles):
    print('\n=== Per-Model Confusion Matrices ===')
    for key, title in zip(model_keys, titles):
        work_dir = os.path.join(main_dir, key)
        csv_path = os.path.join(work_dir, 'epoch1_test_each_class_acc.csv')
        cm = _parse_cm_from_csv(csv_path)
        if cm is None:
            print(f'  {key}: no test CSV found, skipping')
            continue
        n = cm.shape[0]
        labels = _LABELS.get(n, [str(i) for i in range(n)])
        out_path = os.path.join(work_dir, 'confusion_matrix.png')
        save_confusion_matrix_from_cm(cm, out_path, labels=labels, title=title)
        print(f'  Saved {out_path}')


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate summary table and training curves.')
    parser.add_argument('--main-dir', required=True,
                        help='directory containing joint_root_1/, joint_root_14/, bone_root_1/, bone_root_14/')
    arg = parser.parse_args()

    MODEL_KEYS   = ['joint_root_1', 'joint_root_14', 'bone_root_1', 'bone_root_14']
    MODEL_TITLES = ['Joint – Root 1', 'Joint – Root 14', 'Bone – Root 1', 'Bone – Root 14']

    save_per_model_confusion_matrices(arg.main_dir, MODEL_KEYS, MODEL_TITLES)
    save_summary_table(arg.main_dir, MODEL_KEYS, MODEL_TITLES)
    save_training_curves(arg.main_dir, MODEL_KEYS, MODEL_TITLES)
