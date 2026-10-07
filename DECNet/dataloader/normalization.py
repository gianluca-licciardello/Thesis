"""Training-fold-only channel normalization, serialized with each checkpoint."""
import hashlib
import numpy as np


def fit_normalization(dataset):
    keys = sorted({record.DB_index for record in dataset.video_list})
    arrays = [np.asarray(dataset.DB_feature[key], dtype=np.float64) for key in keys]
    if not arrays or any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('Training features must be nonempty and finite')
    stacked = np.concatenate(arrays, axis=1)
    mean = stacked.mean(axis=1, keepdims=True)
    std = stacked.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32),
            'training_keys': keys,
            'training_keys_sha256': hashlib.sha256('\n'.join(keys).encode()).hexdigest()}


def apply_normalization(dataset, stats):
    mean, std = stats['mean'], stats['std']
    dataset.DB_feature = {key: ((np.asarray(value) - mean) / std).astype(np.float32)
                          for key, value in dataset.DB_feature.items()}
