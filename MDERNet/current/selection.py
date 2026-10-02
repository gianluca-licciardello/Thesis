import numpy as np
import engine as worker
FEB=["feb","feb_no_fam","feb_no_fam_fm"]
BGB=["bgb_only_nr","bgb_only_nr_nv"]

def select_pair(records):
    scores={}
    for key in FEB+BGB:
        rows=records[key]
        assert len(rows)==10 and sorted(r['fold'] for r in rows)==list(range(10))
        scores[key]={metric:float(np.mean([r['validation'][metric] for r in rows])) for metric in ['macro_accuracy','macro_f1']}
    rank=lambda k:(scores[k]['macro_accuracy'],scores[k]['macro_f1'])
    bf=max(FEB,key=rank);bb=max(BGB,key=rank)
    return dict(feb=bf,bgb=bb,validation_means=scores,
                rule='mean validation macro-accuracy over ten folds; mean validation macro F1 tie-break; registry order for exact tie',
                kwargs={**worker.VARIANTS[bf][2],**worker.VARIANTS[bb][2],'no_refine':True})

def canonical(kwargs):
    return {'use_fam':True,'use_fm':True,'no_refine':False,'use_visibility':True,'use_bones':True,**kwargs}

def reusable(row,kwargs):
    return row['variant'] in ('mdernet_bgb_nr','combined_best') and canonical(row['kwargs'])==canonical(kwargs)
