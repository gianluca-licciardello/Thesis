"""Summarize saved test metrics as unweighted fold means, without training."""
import argparse,json,csv
from pathlib import Path
from collections import defaultdict

def summarize(folder):
    groups=defaultdict(list)
    for p in sorted(folder.glob('*_fold_[0-9][0-9].json')):
        r=json.loads(p.read_text());groups[r['variant']].append(r)
    rows=[]
    for key,records in groups.items():
        assert len({r['fold'] for r in records})==len(records)
        metrics={m:sum(r['test'][m] for r in records)/len(records) for m in records[0]['test']}
        rows.append(dict(variant=key,folds=len(records),**metrics))
    (folder/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    if rows:
        with (folder/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(summarize(a.folder),indent=2))
