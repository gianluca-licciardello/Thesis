"""Run requested subsets with the current campaign's exact fit/selection code."""
import argparse
import fcntl
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
import selection as fixed

w = fixed.worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['aide_clean', 'aide_balanced', 'ppb_eeg', 'ppb_epq', 'ppb_cluster'])
    parser.add_argument('--split', choices=['independent', 'mixed'], default='independent')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent.parent/'outputs/current')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs and splits without training or writing outputs')
    parser.add_argument('--data-root', type=Path, required=True, help='Exported campaign input bundle')
    args = parser.parse_args()
    bundle = args.data_root.resolve()
    w.AIDE_CACHE = bundle/'aide'
    w.PPB_CACHE = bundle/'ppb'
    w.BACKBONE = bundle/'resnet18_dominik.pth'
    if w.digest(w.BACKBONE) != '734341508e3ddbcd181e40da0173cf2c77cd0de3bc2455715cc1b7f42a63a6f6':
        raise ValueError('Expected the current campaign Dominik backbone')
    original = {p.name:w.digest(p) for p in Path(__file__).resolve().parent.glob('*.py')}
    aide = args.dataset.startswith('aide')
    key = args.dataset if aide else args.dataset+'_'+args.split
    if aide:
        df = pd.read_csv(w.AIDE_CACHE/'labels.csv', dtype={'clip_id': str})
        subset_path = bundle/('aide_clean_keypoints_subset.json' if args.dataset == 'aide_clean' else 'aide_balanced_subset.json')
        kept = set(json.loads(subset_path.read_text())['kept_clip_ids'])
        folds = [te for _, te in StratifiedKFold(10, shuffle=True, random_state=42).split(df, df.discrete_label)]
        splits = []
        for i in range(10):
            tr = np.sort(np.concatenate([folds[j] for j in range(10) if j not in (i, (i+1)%10)]))
            tr = tr[df.iloc[tr].clip_id.isin(kept).to_numpy()]
            splits.append((tr, folds[(i+1)%10], folds[i]))
    else:
        subset_path = bundle/'subject_subsets.json'
        df, splits = w.build_kfold_splits(preproc_dir=str(w.PPB_CACHE), k_folds=10,
            mix_subjects=args.split == 'mixed', with_validation=True, random_state=42,
            downsample_train=True, subsets_path=str(subset_path),
            subset_name={'ppb_eeg':'EEG', 'ppb_epq':'EPQ', 'ppb_cluster':'Cluster'}[args.dataset])
    body_path = bundle/('aide_body.npy' if aide else 'ppb_body.npy')
    body = np.load(body_path, mmap_mode='r')
    assert body.shape == (len(df), 30, 15, 4), body.shape
    data_manifest = json.loads((bundle/'data_manifest.json').read_text())
    assert w.digest(body_path) == data_manifest['hashes'][body_path.relative_to(bundle).as_posix()], 'Body cache changed'
    cache = w.AIDE_CACHE if aide else w.PPB_CACHE
    label_path = cache/'labels.csv'
    assert w.digest(label_path) == data_manifest['hashes'][label_path.relative_to(bundle).as_posix()], 'Labels changed'
    assert w.digest(subset_path) == data_manifest['hashes'][subset_path.relative_to(bundle).as_posix()], 'Subset definition changed'
    for row in df.itertuples():
        clip = row.clip_id if aide else f'{row.participant}_{row.emotion_code}'
        face = cache/'faces'/(clip+'.npy')
        assert w.digest(face) == data_manifest['hashes'][face.relative_to(bundle).as_posix()], 'Face cache changed: '+face.relative_to(bundle).as_posix()
    for tr, va, te in splits:
        assert len(tr) > 1, 'Training fold too small'
        assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
        if not aide and args.split == 'independent':
            groups = [set(df.iloc[ix].participant) for ix in (tr, va, te)]
            assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    print(key, 'train/validation/test sizes:', [(len(tr),len(va),len(te)) for tr,va,te in splits], flush=True)
    if any(len(tr) < 64 for tr, _, _ in splits):
        print('Small training folds: retain their partial batch instead of dropping all samples', flush=True)
    if args.check_only:
        print('PASS: source, backbone, data, subset and split checks; no training launched')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('The current campaign training code requires CUDA')
    out = args.output_root.resolve()/key
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = dict(dataset=key, original=original, subset_sha256=w.digest(subset_path),
            data_manifest_sha256=w.digest(bundle/'data_manifest.json'),
            launcher_sha256=w.digest(Path(__file__)), selection_sha256=w.digest(Path(fixed.__file__)),
            settings=dict(epochs=50, lr=.01 if aide else .0001, batch_size=64, seed=42, dropout=0., folds=10),
            policy='Training-only subset; configuration-first; fixed global validation-selected fusion; retain partial batch when training fold has fewer than 64 clips')
        path = out/'manifest.json'
        if path.exists():
            assert json.loads(path.read_text()) == manifest, 'Run settings changed; use a new --output-root'
        else:
            w.save(path, manifest)
        dest = out/body_path.name
        if not dest.exists(): dest.symlink_to(body_path)
        # Keep the existing loader settings except when drop_last would erase a fold.
        original_loader = w.DataLoader
        def make_loader(dataset, *a, **kw):
            if kw.get('drop_last') and len(dataset) < kw.get('batch_size', 64):
                kw['drop_last'] = False
            return original_loader(dataset, *a, **kw)
        w.DataLoader = make_loader
        w.HERE = out
        w.report = lambda: None
        w.save(out/'runs'/key/'splits.json', [{n:ix.tolist() for n,ix in zip(['train','validation','test'],sp)} for sp in splits])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        try:
            for variant, (_, kind, kwargs) in w.VARIANTS.items():
                for fold, split in enumerate(splits):
                    w.fit(key, fold, variant, kind, kwargs, df, split)
            folder = out/'runs'/key
            records = {v:[json.loads((folder/f'{v}_fold_{i:02d}.json').read_text()) for i in range(10)] for v in fixed.FEB+fixed.BGB}
            chosen = fixed.select_pair(records)
            w.save(folder/'selection.json', chosen)
            for fold, split in enumerate(splits):
                path = folder/f'fixed_best_fold_{fold:02d}.json'
                if path.exists(): continue
                src = folder/f'mdernet_bgb_nr_fold_{fold:02d}.json'
                candidate = json.loads(src.read_text())
                if fixed.reusable(candidate, chosen['kwargs']):
                    result = dict(candidate, variant='fixed_best', reused_from=str(src))
                else:
                    result = w.fit(key, fold, 'fixed_best', 'fusion', chosen['kwargs'], df, split)
                result.update(selected_feb=chosen['feb'], selected_bgb=chosen['bgb'], selection_scope='fixed global mean validation selection')
                w.save(path, result)
            w.state('complete', experiment=key)
        except Exception as exc:
            w.state('failed', experiment=key, error=repr(exc))
            raise


if __name__ == '__main__':
    main()
