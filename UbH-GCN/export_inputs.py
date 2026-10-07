"""Export only prepared arrays/labels referenced by the saved run configurations."""
import argparse,hashlib,json,shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',required=True,type=Path);p.add_argument('--destination',required=True,type=Path);a=p.parse_args()
 files=set()
 for cfg in (ROOT/'configs').glob('*/*.yaml'):
  c=yaml.safe_load(cfg.read_text())
  for k in ['train_feeder_args','eval_feeder_args','test_feeder_args']:
   for field in ['data_path','label_path']:
    rel=Path(c[k][field]);assert not rel.is_absolute() and '..' not in rel.parts;files.add(rel)
 for rel in files:assert (a.source_root/rel).is_file(),str(rel)
 a.destination.mkdir(parents=True,exist_ok=False);hashes={}
 for rel in sorted(files):
  dest=a.destination/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.source_root/rel,dest)
  sha=digest(a.source_root/rel);assert digest(dest)==sha;hashes[rel.as_posix()]=sha
 (a.destination/'manifest.json').write_text(json.dumps({'sha256':hashes},indent=2)+'\n')
 print(f'Exported and verified {len(files)} prepared input files to {a.destination}')
if __name__=='__main__':main()
