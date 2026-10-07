import argparse
import pickle
import os

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import math
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from plot_utils import save_confusion_matrix as _plot_cm

EMOTION_LABELS_AIDE    = ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger']
EMOTION_LABELS_PPB_EMO = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def show_important_joints(result):
        first_sum = np.sum(result[:,:,0], axis=0)
        first_index = np.argsort(-first_sum) + 1
        print('Weights of all joints:')
        print(first_sum)
        print('')
        print('Most important joints:')
        print(first_index)
        print('')


def _run_ensemble(score_lists, label, norm):
    """Fuse score_lists by summing L2-normalised vectors; return (y_true, y_pred)."""
    y_true, y_pred = [], []
    for i in range(len(label)):
        r = sum(norm(np.array(sl[i][1])) for sl in score_lists)
        y_true.append(int(label[i]))
        y_pred.append(int(np.argmax(r)))
    return y_true, y_pred


def _report(y_true, y_pred, title, main_dir, out_name, emotion_labels):
    """Print rounded metrics and save a confusion matrix image.

    Reports both the "normal" (overall) accuracy / weighted F1 and their macro
    counterparts, matching report.py's definitions for the single-model tables:
      - Macro accuracy = mean of per-class recall (a.k.a. balanced accuracy).
      - Macro F1        = unweighted mean of per-class F1 (equal weight per class).
    """
    acc = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    F1  = f1_score(y_true, y_pred, average='weighted')
    macro_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average='macro')
    cm  = confusion_matrix(y_true, y_pred)

    print(f'\n--- {title} ---')
    print('Acc: {:.2f}%'.format(acc * 100))
    print('Macro Acc: {:.2f}%'.format(macro_acc * 100))
    print('Weighted F1: {:.2f}%'.format(F1 * 100))
    print('Macro F1: {:.2f}%'.format(macro_f1 * 100))
    print('Confusion Matrix:')
    print(cm)

    out_path = os.path.join(main_dir, out_name)
    _plot_cm(y_true, y_pred, out_path=out_path, labels=emotion_labels, title=title)
    print(f'Confusion matrix saved to {out_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'AIDE', 'PPB_Emo'},
                        help='the work folder for storing results')
    parser.add_argument('--main-dir',
                        required=True,
                        help='directory that contains the modality score folders')
    parser.add_argument('--label-path',
                        default=None,
                        help='path to the dataset label pickle file')
    parser.add_argument('--root-1',
                        type=str2bool,
                        default=True)
    parser.add_argument('--root-14',
                        type=str2bool,
                        default=True)
    parser.add_argument('--best-epoch',
                        type=int,
                        help='')

    arg = parser.parse_args()

    dataset = arg.dataset
    _repo_root = Path(__file__).resolve().parent
    if arg.dataset == 'AIDE':
        _default_label = _repo_root / 'data' / 'AIDE' / 'processed_data' / 'test_label.pkl'
        _emotion_labels = EMOTION_LABELS_AIDE
    elif arg.dataset == 'PPB_Emo':
        _default_label = _repo_root / 'data' / 'PPB_Emo' / 'processed_data' / 'test_label.pkl'
        _emotion_labels = EMOTION_LABELS_PPB_EMO
    else:
        raise NotImplementedError

    label_path = Path(arg.label_path) if arg.label_path else _default_label
    if not label_path.exists():
        raise FileNotFoundError(
            f'Could not find the label file at {label_path}. '
            'Pass --label-path if your dataset is stored elsewhere.'
        )
    with open(label_path, 'rb') as f:
        label = pickle.load(f)

    norm = lambda x: x / np.linalg.norm(x)

    dir_cnt = 0

    if arg.root_1:
        with open(os.path.join(arg.main_dir, 'joint_root_1/', 'epoch1_test_score.pkl'), 'rb') as r1:
            r1 = list(pickle.load(r1).items())
        with open(os.path.join(arg.main_dir, 'bone_root_1/', 'epoch1_test_score.pkl'), 'rb') as r2:
            r2 = list(pickle.load(r2).items())
        dir_cnt += 2

    if arg.root_14:
        with open(os.path.join(arg.main_dir, 'joint_root_14/' 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
        with open(os.path.join(arg.main_dir, 'bone_root_14/', 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())
        dir_cnt += 2

    if dir_cnt == 4:
        # 2-way sub-ensemble: Joint + Bone, Root 1
        yt1, yp1 = _run_ensemble([r1, r2], label, norm)
        _report(yt1, yp1,
                title='2-way Ensemble (Joint + Bone, Root 1)',
                main_dir=arg.main_dir,
                out_name='confusion_matrix_ensemble_root1.png',
                emotion_labels=_emotion_labels)

        # 2-way sub-ensemble: Joint + Bone, Root 14
        yt14, yp14 = _run_ensemble([r3, r4], label, norm)
        _report(yt14, yp14,
                title='2-way Ensemble (Joint + Bone, Root 14)',
                main_dir=arg.main_dir,
                out_name='confusion_matrix_ensemble_root14.png',
                emotion_labels=_emotion_labels)

        # 4-way ensemble
        yt4, yp4 = _run_ensemble([r1, r2, r3, r4], label, norm)
        _report(yt4, yp4,
                title='4-way Ensemble (Joint + Bone, Root 1 & Root 14)',
                main_dir=arg.main_dir,
                out_name='confusion_matrix_ensemble.png',
                emotion_labels=_emotion_labels)

    elif dir_cnt == 2:
        if arg.root_1:
            score_lists = [r1, r2]
            title = '2-way Ensemble (Joint + Bone, Root 1)'
        else:
            score_lists = [r3, r4]
            title = '2-way Ensemble (Joint + Bone, Root 14)'

        yt, yp = _run_ensemble(score_lists, label, norm)
        _report(yt, yp,
                title=title,
                main_dir=arg.main_dir,
                out_name='confusion_matrix_ensemble.png',
                emotion_labels=_emotion_labels)
