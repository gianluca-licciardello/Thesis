"""Export exact campaign inputs as a portable folder; no results or raw videos."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True, help='Original MDERNet directory')
    p.add_argument('--subset-root', type=Path, required=True, help='Directory containing the three subset JSONs')
    p.add_argument('--destination', type=Path, required=True, help='New portable input directory (about 4.6 GiB)')
    args = p.parse_args()
    src = args.source_root.resolve()
    campaign = src/'campaigns/validation_20260930'
    old = json.loads((campaign/'data_manifest.json').read_text())['hashes']
    expected_backbone = json.loads((campaign/'manifest.json').read_text())['backbone_sha256']
    files = [(src/'resnet18_dominik.pth', 'resnet18_dominik.pth')]
    for name in ['aide_body.npy', 'ppb_body.npy']:
        files.append((campaign/name, name))
    for name in ['aide_clean_keypoints_subset.json', 'aide_balanced_subset.json', 'subject_subsets.json']:
        files.append((args.subset_root/name, name))
    for name, cache in [('aide',src/'results_legacy/aide_balanced_dominik_e50_lr001_interpolated/preprocessed'), ('ppb',src/'preprocessed')]:
        files.append((cache/'labels.csv', name+'/labels.csv'))
        files.extend((f,name+'/faces/'+f.name) for f in sorted((cache/'faces').glob('*.npy')))
    if args.destination.exists():
        raise FileExistsError('Choose a new destination; existing bundles are never overwritten')
    args.destination.mkdir(parents=True)
    hashes = {}
    for original, relative in files:
        sha = digest(original)
        expected = expected_backbone if relative == 'resnet18_dominik.pth' else old.get(str(original))
        if expected is not None and sha != expected:
            raise ValueError('Campaign input changed: '+str(original))
        target = args.destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if digest(target) != sha:
            raise IOError('Copy verification failed: '+str(target))
        hashes[relative] = sha
    (args.destination/'data_manifest.json').write_text(json.dumps({'hashes':hashes}, indent=2)+'\n')
    print(f'Exported and verified {len(files)} files to {args.destination}')


if __name__ == '__main__':
    main()
