#!/usr/bin/env python3
"""C69 companion: simplest H+M joint-conditioning conditional RealNVP.
Uses ONLY original C64 1600 train/240 val pairs. The 240 original C64 test
pairs are kept for independent joint conditional log score, without any PDE
calls or explicit H/M expert communication. All test samples are evaluated.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor,cho_solve
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(REPO/'flow'/'c64_legacy'))
sys.path.insert(0,str(REPO/'solver'))
import c64_conditional_flow as G
import biot_mechanism_audit as B
import c32_linked_storage_audit as C

torch.set_num_threads(1)
with np.load(ROOT/'joint_dataset.npz') as d: data={k:d[k] for k in d.files}

def init_joint(d,mode):
    rows=np.arange(90)
    base=d['base'];sigma=d['sigma'];basey=d['base_y']
    _,J,_=C.forward(base[:256],base[256:],.7,load='pulse',jac=True)
    A=J[rows,:512]@B.TM/sigma[rows,None]
    Z=np.eye(len(rows))+A@A.T
    ch=cho_factor(Z,lower=True,check_finite=False)
    def feat(y):
        r=(y[:,rows]-basey[rows])/sigma[rows]
        mu=(r@cho_solve(ch,A,check_finite=False)).astype('float32')
        return mu,(r/10.).astype('float32')
    return A,feat,np.linalg.slogdet(Z)[1],rows

def logscore_for(mode, ckpt):
    if mode=='J':
        A,feat,logdet,rows=init_joint(data,mode)
    else:
        A,feat,logdet,rows=G.init_physics(data,mode)
    mu,ctx=feat(data['y'][-240:])
    z=torch.tensor(data['z'][-240:],dtype=torch.float32)
    tmu=torch.tensor(mu);tctx=torch.tensor(ctx)
    net=G.CondFlow(len(rows))
    net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True))
    net.eval()
    with torch.no_grad():
        z0,ld=net.inverse(z,tctx,tmu)
        dz=z0-tmu
        mat=torch.tensor(A,dtype=torch.float32)
        # Neg log q, including logdet normalization, excluding universal 512D constant.
        q_nll=(.5*dz.square().sum(1)+.5*(dz@mat.T).square().sum(1)-ld-.5*logdet).numpy()
        d=z-tmu
        base_nll=(.5*d.square().sum(1)+.5*(d@mat.T).square().sum(1)-.5*logdet).numpy()
    return q_nll.astype(float),base_nll.astype(float),{'nobs':len(rows),'logdet':float(logdet)}

if __name__=='__main__':
    original=G.init_physics
    G.init_physics=init_joint
    out=ROOT/'joint_flow';out.mkdir(exist_ok=True)
    result=G.train_one(data,'J',out,seed=641133,train_n=1600,val_n=240,epochs=32,batch=80)
    G.init_physics=original
    nll={};meta={}
    for mode in ('H','M','J'):
        checkpoint=out/'J_conditional_flow.pt' if mode=='J' else SRC/'flow_results'/f'{mode}_conditional_flow.pt'
        nll[f'{mode}_Flow'],nll[f'{mode}_Gaussian'],meta[mode]=logscore_for(mode,checkpoint)
    t=pd.DataFrame({'index':range(240),**nll});t.to_csv(out/'all240_proper_logscores.csv',index=False)
    rng=np.random.default_rng(641134)
    comparisons=[('J_Flow','J_Gaussian'),('J_Flow','M_Flow'),('J_Flow','H_Flow'),('M_Flow','M_Gaussian'),('H_Flow','H_Gaussian')]
    stats=[]
    for a,b in comparisons:
        delta=(t[a]-t[b]).values
        ids=rng.integers(0,len(delta),(20000,len(delta)))
        ci=np.quantile(delta[ids].mean(1),[.025,.975])
        stats.append({'candidate':a,'reference':b,'n':len(delta),'mean_delta_nll':float(delta.mean()),
        'ci_low':float(ci[0]),'ci_high':float(ci[1]),'better_fraction':float((delta<0).mean())})
    pd.DataFrame(stats).to_csv(out/'paired_comparisons.csv',index=False)
    (out/'audit.json').write_text(json.dumps({'train_pairs':1600,'val_pairs':240,'test_pairs':240,
        'source_dataset':'C64 original unchanged','test_used_for_training':False,
        'checkpoint_best_epoch':result['best_epoch'],'n_parameters':result['n_parameters'],'normalization':meta,
        'comparison_statistics':stats},indent=2))
    print('JOINT',json.dumps(stats,indent=2),flush=True)
