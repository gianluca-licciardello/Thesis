import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import fcntl
import hashlib
import json
import random
import resource
from contextlib import ExitStack
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


from config import Config
from model import FacialExpressionBranch, MDERNet, BGFEM, DM
from evaluate import combined_loss, compute_ccc
from dataset import build_kfold_splits

HERE = Path(__file__).resolve().parent
ROOT = HERE
AIDE_CACHE = HERE
PPB_CACHE = HERE
BACKBONE = HERE / "resnet18_dominik.pth"
VARIANTS = {
    'feb': ('FEB', 'feb', dict(use_fam=True, use_fm=True)),
    'feb_no_fam': ('FEB (w/o FAM)', 'feb', dict(use_fam=False, use_fm=True)),
    'feb_no_fam_fm': ('FEB (w/o FAM/FM)', 'feb', dict(use_fam=False, use_fm=False)),
    'bgb_only_nr': ('BGB (w/o refine)', 'body', dict(use_visibility=True, use_bones=True)),
    'bgb_only_nr_nv': ('BGB (w/o refine/visibility)', 'body', dict(use_visibility=False, use_bones=True)),
    'mdernet_bgb': ('MDERNet (FEB + BGB)', 'fusion', dict(no_refine=False, use_visibility=True, use_bones=True)),
    'mdernet_bgb_nr': ('MDERNet (FEB + BGB w/o refine)', 'fusion', dict(no_refine=True, use_visibility=True, use_bones=True)),
}
NAMES = {k:v[0] for k,v in VARIANTS.items()}
NAMES['combined_best'] = 'Validation-selected FEB + BGB fusion'


def report():
    pass

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(obj if isinstance(obj,str) else json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def state(status, **kw):
    save(HERE/'status.json', dict(state=status,updated=datetime.now().astimezone().isoformat(timespec='seconds'),**kw))

def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

class Samples(Dataset):
    def __init__(self,df,idx,aide,body_only=False):
        self.df=df;self.idx=idx;self.aide=aide;self.body_only=body_only
        self.body=np.load(HERE/('aide_body.npy' if aide else 'ppb_body.npy'),mmap_mode='r')
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        ix=self.idx[i];row=self.df.iloc[ix]
        clip=row.clip_id if self.aide else f'{row.participant}_{row.emotion_code}'
        cache=AIDE_CACHE if self.aide else PPB_CACHE
        face=torch.zeros(1) if self.body_only else torch.from_numpy(np.load(cache/'faces'/(clip+'.npy'))).unsqueeze(1)
        dims=np.zeros(3,np.float32) if self.aide else np.array([row.valence,row.arousal,row.dominance],np.float32)
        return face,torch.from_numpy(self.body[ix].copy()),int(row.discrete_label),torch.from_numpy(dims)

class BodyOnly(nn.Module):
    """Exactly the no-refinement body path, without unused facial computation."""
    def __init__(self,classes,**kwargs):
        super().__init__();self.kwargs=kwargs
        self.secondary=BGFEM(dropout=0.,**kwargs); self.dm=DM(num_classes=classes)
    def forward(self,faces,body):
        xyz=body[...,:3];v=body[...,3:4];parts=[xyz.flatten(2)]
        if self.kwargs['use_visibility']:parts.append(v.flatten(2))
        if self.kwargs['use_bones']:
            a,b=np.array(Config.BGB_BONE_PAIRS).T
            parts.append(((xyz[:,:,b]-xyz[:,:,a])*v[:,:,a]*v[:,:,b]).flatten(2))
        features=self.secondary(torch.cat(parts,-1).flatten(1))
        return self.dm(torch.zeros(len(body),512,device=body.device),features)

def make_model(kind,kwargs,aide):
    classes=5 if aide else 7
    if kind=='body':model=BodyOnly(classes,**kwargs)
    else:
        cls=FacialExpressionBranch if kind=='feb' else MDERNet
        more={} if kind=='feb' else dict(branch='body')
        model=cls(num_classes=classes,pretrained_path=str(BACKBONE),**more,**kwargs)
    for m in model.modules():
        if isinstance(m,nn.Dropout):m.p=0.
    return model

def infer(model,kind,face,body):
    return model(face)[:2] if kind=='feb' else model(face,body)[:2]

def loss_fn(logits,dims,y,target,aide):
    return combined_loss(logits,dims,y,target,lambda_mse=0. if aide else .1,lambda_ccc=0. if aide else 1.,lambda_f1=1. if aide else 0.)[0]

@torch.no_grad()
def evaluate(model,kind,loader,aide):
    model.eval();ys=[];ps=[];probs=[];dtrue=[];dpred=[];total=0.;count=0
    for face,body,y,dims in loader:
        face,body,y,dims=[v.cuda(non_blocking=True) for v in (face,body,y,dims)]
        logits,pdim=infer(model,kind,face,body);loss=loss_fn(logits,pdim,y,dims,aide)
        assert torch.isfinite(loss) and torch.isfinite(logits).all()
        total+=float(loss)*len(y);count+=len(y)
        ys.extend(y.cpu().tolist());ps.extend(logits.argmax(1).cpu().tolist());probs.extend(logits.softmax(-1).cpu().tolist())
        dtrue.extend(dims.cpu().tolist());dpred.extend(pdim.cpu().tolist())
    metrics=dict(accuracy=float(accuracy_score(ys,ps)),macro_accuracy=float(balanced_accuracy_score(ys,ps)),
                 macro_f1=float(f1_score(ys,ps,labels=range(5 if aide else 7),average='macro',zero_division=0)),
                 weighted_f1=float(f1_score(ys,ps,average='weighted',zero_division=0)),loss=total/count)
    if not aide:metrics.update(mse=float(np.mean((np.array(dtrue)-np.array(dpred))**2)),ccc=compute_ccc(np.array(dtrue),np.array(dpred)))
    return metrics,dict(true=ys,pred=ps,probs=probs,dim_true=dtrue,dim_pred=dpred,confusion_matrix=confusion_matrix(ys,ps,labels=range(5 if aide else 7)).tolist())

def fit(key,fold,variant,kind,kwargs,df,split):
    # Pin-memory + persistent-worker loaders can retain descriptor cleanup
    # registrations across many fits. Keep worker RNG lifetime unchanged, but
    # disable pinning and explicitly terminate each fit's workers, also on error.
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    target=65536 if hard==resource.RLIM_INFINITY else min(65536,hard)
    if soft<target:resource.setrlimit(resource.RLIMIT_NOFILE,(target,hard))
    with ExitStack() as resources:
        return _fit(key,fold,variant,kind,kwargs,df,split,resources)

def _close_loader(loader):
    iterator=getattr(loader,'_iterator',None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator=None

def _fit(key,fold,variant,kind,kwargs,df,split,resources):
    dest=HERE/'runs'/key;result_path=dest/f'{variant}_fold_{fold:02d}.json'
    if result_path.exists():return json.loads(result_path.read_text())
    aide=key.startswith('aide');seed();model=make_model(kind,kwargs,aide).cuda()
    gen=torch.Generator().manual_seed(42)
    loaders=[DataLoader(Samples(df,ix,aide,kind=='body'),batch_size=64,shuffle=i==0,drop_last=i==0,
                        num_workers=4,pin_memory=False,persistent_workers=True,generator=gen if i==0 else None) for i,ix in enumerate(split)]
    for loader in loaders:resources.callback(_close_loader,loader)
    opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,nesterov=True,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=5,eta_min=1e-5)
    best=(-1.,-1.);history=[];ckpt=dest/f'{variant}_fold_{fold:02d}_best.pt'
    for epoch in range(1,51):
        state('running',experiment=key,variant=variant,fold=fold,epoch=epoch);report()
        model.train();total=0.;seen=0;correct=0
        for face,body,y,dims in loaders[0]:
            face,body,y,dims=[v.cuda(non_blocking=True) for v in (face,body,y,dims)]
            opt.zero_grad(set_to_none=True);logits,pdim=infer(model,kind,face,body);loss=loss_fn(logits,pdim,y,dims,aide)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('Nonfinite gradients')
            opt.step();total+=float(loss.detach())*len(y);seen+=len(y);correct+=int((logits.argmax(1)==y).sum())
        sched.step();val,_=evaluate(model,kind,loaders[1],aide)
        history.append(dict(epoch=epoch,train_loss=total/seen,train_accuracy=correct/seen,validation=val))
        rank=(val['macro_accuracy'],val['macro_f1'])
        if rank>best:
            best=rank;best_epoch=epoch;best_val=val
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},ckpt)
        save(dest/f'{variant}_fold_{fold:02d}_history.json',history)
        print(key,variant,'fold',fold,'epoch',epoch,'loss',total/seen,'val_macro_accuracy',val['macro_accuracy'],flush=True)
    model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True))
    test,pred=evaluate(model,kind,loaders[2],aide)
    result=dict(variant=variant,fold=fold,seed=42,best_epoch=best_epoch,validation=best_val,test=test,predictions=pred,kwargs=kwargs,checkpoint=str(ckpt.relative_to(HERE)))
    save(result_path,result);report()
    del model,loaders,opt,sched;torch.cuda.empty_cache()
    return result
