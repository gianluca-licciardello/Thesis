"""Train saved work_dir configurations, then generate ensembles and reports."""
import argparse,json,pickle,subprocess,sys
from pathlib import Path
import numpy as np
import yaml
from export_inputs import digest
ROOT=Path(__file__).resolve().parent

def main():
 groups=sorted(p.name for p in (ROOT/'configs').iterdir() if p.is_dir())
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True,type=Path);p.add_argument('--output-root',type=Path,default=ROOT/'work_dir');p.add_argument('--group',choices=groups);p.add_argument('--check-only',action='store_true');p.add_argument('--device',type=int,default=0);a=p.parse_args()
 bundle=a.data_root.resolve();manifest=json.loads((bundle/'manifest.json').read_text())['sha256'];checked=set();plans=[]
 for group in groups:
  if a.group and group!=a.group:continue
  configs=[]
  for cfg in sorted((ROOT/'configs'/group).glob('*.yaml')):
   c=yaml.safe_load(cfg.read_text());assert c['weights'] is None and c['phase']=='train'
   for k in ['train_feeder_args','eval_feeder_args','test_feeder_args']:
    args=c[k]
    for field in ['data_path','label_path']:
     rel=Path(args[field]);path=bundle/rel
     if rel.as_posix() not in checked:
      assert digest(path)==manifest[rel.as_posix()],str(path);checked.add(rel.as_posix())
     args[field]=str(path)
    x=np.load(args['data_path'],mmap_mode='r')
    with open(args['label_path'],'rb') as f:y=np.asarray(pickle.load(f))
    assert len(x)==len(y) and x.ndim==5 and x.shape[1]==c['model_args'].get('in_channels',3) and x.shape[3]==c['model_args']['num_point']
    assert np.isfinite(x).all() and np.all((y>=0)&(y<c['model_args']['num_class']))
   c['work_dir']=str(a.output_root.resolve()/group/cfg.stem);c['device']=[a.device];configs.append((cfg.stem,c))
  assert {name for name,c in configs}=={'joint_root_1','joint_root_14','bone_root_1','bone_root_14'}
  plans.append((group,configs));print('Validated',group,flush=True)
 if a.check_only:
  print(f'PASS: {sum(len(c) for g,c in plans)} configurations; no training started');return
 # Refuse existing group output rather than letting the original trainer overwrite it.
 for group,configs in plans:
  if (a.output_root.resolve()/group).exists():raise FileExistsError(f'{group}: choose a new --output-root')
 for group,configs in plans:
  folder=a.output_root.resolve()/group;folder.mkdir(parents=True)
  runtime=folder/'input_configs';runtime.mkdir()
  for stream,c in configs:
   path=runtime/(stream+'.yaml');c['config']=str(path);path.write_text(yaml.safe_dump(c,sort_keys=False))
   subprocess.run([sys.executable,str(ROOT/'main.py'),'--config',str(path)],cwd=ROOT,check=True)
  label=configs[0][1]['test_feeder_args']['label_path']
  assert all(c['test_feeder_args']['label_path']==label for _,c in configs)
  subprocess.run([sys.executable,str(ROOT/'ensemble.py'),'--dataset','AIDE' if group.startswith('aide') else 'PPB_Emo','--main-dir',str(folder),'--label-path',label],cwd=ROOT,check=True)
  subprocess.run([sys.executable,str(ROOT/'report.py'),'--main-dir',str(folder)],cwd=ROOT,check=True)
if __name__=='__main__':main()
