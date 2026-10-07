"""Historical DECNet experiment definitions and portable launch utility."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def experiments():
    aide='AIDE_incar_V_DB_rgb_112_3s';ppb='PPB_CIR_V_DB_rgb_112_3s_stride3s'
    for tag,suffix in [('full',''),('balanced','_balanced_dstrain'),('clean','_clean_kp_dstrain')]:
        yield dict(name='aide_'+tag,annotation=aide+suffix,features='aide_db_features.pkl',channels=45,dataset='AIDE',emotions='Anger,Anxiety,Happiness,Peace,Weariness',classes=5)
    for subset,suffix in [('full',''),('cluster','_cluster'),('epq','_epq'),('eeg','_eeg'),('clean','_cleankp')]:
        for mixed in [False,True]:
            for modality,feature,ch in [('driving','db_features_3s_stride3s.pkl',8),('fast_sam3d','ppb_kp_features_3s_stride3s.pkl',45)]:
                yield dict(name=f'ppb_{subset}_{modality}_'+('mixed' if mixed else 'independent'),annotation=ppb+suffix+('_mixed' if mixed else ''),features=feature,channels=ch,dataset='PPB',emotions='AD,DD,FD,HD,ND,SAD,SD',classes=7)

def annotations(e):
    return [e['annotation']+'_test_fold0.txt']+[e['annotation']+f'_cv{k}_{part}_fold0.txt' for k in range(5) for part in ['train','val']]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--only',choices=[e['name'] for e in experiments()]);p.add_argument('--check-only',action='store_true');p.add_argument('--results-dir',type=Path,default=ROOT/'results');a=p.parse_args();bundle=a.data_root.resolve()
    for e in experiments():
        if a.only and e['name']!=a.only:continue
        assert (bundle/'data'/e['features']).is_file(), e['features']
        generated={}
        for name in annotations(e):
            rows=[]
            for line in (bundle/'annotation'/name).read_text().splitlines():
                parts=line.split();frame=bundle/parts[0]
                assert frame.is_dir(),str(frame)
                assert len(list(frame.glob('*.jpg')))>=int(parts[1]), str(frame)
                parts[0]=str(frame);rows.append(' '.join(parts))
            generated[name]='\n'.join(rows)+'\n'
        print('Validated',e['name'],flush=True)
        if a.check_only:continue
        ann=a.results_dir.resolve()/'input_annotations';ann.mkdir(parents=True,exist_ok=True)
        for name,text in generated.items():(ann/name).write_text(text)
        cmd=[sys.executable,str(ROOT/'run_experiment.py'),'--run_name',e['name'],'--annotation_prefix',str(ann/e['annotation']),'--db_pkl_path',str(bundle/'data'/e['features']),'--dataset',e['dataset'],'--db_in_channels',str(e['channels']),'--num_classes',str(e['classes']),'--emotions',e['emotions'],'--outer_fold','0','--n_inner_folds','5','--contrast','V-DB','-t','3','--epochs','50','-b','32','-j','8','--lr','0.01','--momentum','0.9','--wd','0.0001','--noise_A','0.1','--results_dir',str(a.results_dir.resolve())]
        subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__':main()
