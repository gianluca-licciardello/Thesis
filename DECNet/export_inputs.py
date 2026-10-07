"""Export prepared annotations, frames and feature pickles; never include results."""
import argparse,hashlib,json,shutil
from pathlib import Path
from historical_suite import experiments,annotations

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--destination',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    source=a.source_root.resolve();files={};dirs={}
    for e in experiments():
        feature=source/'data'/e['features'];assert feature.is_file(),str(feature)
        for name in annotations(e):
            rows=[]
            for line in (source/'annotation'/name).read_text().splitlines():
                parts=line.split();old=Path(parts[0]);assert old.is_dir(),str(old)
                relative='frames/'+hashlib.sha256(str(old).encode()).hexdigest()[:20]
                dirs[relative]=old;parts[0]=relative;rows.append(' '.join(parts))
            files[name]='\n'.join(rows)+'\n'
    # Confirm frame coverage and feature-key coverage before copying anything.
    import pickle
    for e in experiments():
        with (source/'data'/e['features']).open('rb') as f:features=pickle.load(f)
        for name in annotations(e):
            for line in files[name].splitlines():
                row=line.split();assert row[3] in features,(name,row[3]);assert features[row[3]].shape[0]==e['channels']
    for rel,old in dirs.items():
        assert list(old.glob('*.jpg')),str(old)
    print(f'Validated {len(files)} annotations, {len(dirs)} frame directories and 3 feature pickles')
    if a.check_only:return
    if a.destination is None:p.error('--destination is required for export')
    a.destination.mkdir(parents=True,exist_ok=False)
    (a.destination/'annotation').mkdir();(a.destination/'data').mkdir()
    for name,text in files.items():(a.destination/'annotation'/name).write_text(text)
    for name in sorted({e['features'] for e in experiments()}):shutil.copy2(source/'data'/name,a.destination/'data'/name)
    for rel,old in dirs.items():shutil.copytree(old,a.destination/rel)
    (a.destination/'INPUTS.json').write_text(json.dumps({'source_root':str(source),'annotations':len(files),'frame_directories':len(dirs),'note':'Export of available prepared inputs; historical identity is not certified'},indent=2)+'\n')
    print('Exported to',a.destination)
if __name__=='__main__':main()
