#!/usr/bin/env python3
"""Development-only tuning of strong full-coordinate spline controls (no frozen test access)."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import time,json,sys
from pathlib import Path
import numpy as np,torch,pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import c74_p5_strong_baselines as B
F=B.P4

def fit(mode,hidden,seed,lr,epochs=12):
 torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
 net=B.FullCoordinateSpline(hidden)
 optim=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=5e-5)
 zz,mu,ctx,ar=F.DATA_Z,F.DATA_MU,F.DATA_CTX,F.AR
 best=float('inf');epbest=0;hist=[];start=time.time()
 ck=P/'checkpoints'/f'{mode}_seed{seed}.pt'
 for ep in range(1,epochs+1):
  net.train()
  for ids in np.array_split(rng.permutation(1600),20):
   optim.zero_grad()
   x=zz[ids];mm=mu[ids];c=ctx[ids]
   u,ld=net.inverse(x,c,mm);d=u-mm
   loss=(.5*d.square().sum(1)+.5*(d@ar.T).square().sum(1)-ld-.5*F.DG.CF.logdet).mean()+.3*(u-x).square().mean()
   loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.5);optim.step()
  net.eval()
  with torch.no_grad():vl=float(B.nll(net,zz[1600:1840],mu[1600:1840],ctx[1600:1840],ar,F.DG.CF.logdet).mean())
  hist.append({'epoch':ep,'val_nll':vl});print('DEV',mode,seed,ep,round(vl,4),flush=True)
  if vl<best-1e-5:best=vl;epbest=ep;torch.save(net.state_dict(),ck)
 row={'mode':mode,'seed':seed,'parameters':sum(p.numel() for p in net.parameters()),'lr':lr,'epochs':epochs,'best_epoch':epbest,'val_nll':best,'seconds':time.time()-start}
 (P/'checkpoints'/f'{mode}_seed{seed}_train.json').write_text(json.dumps({'summary':row,'history':hist},indent=2))
 print('TUNED',json.dumps(row),flush=True)
 return row
if __name__=='__main__':
 seed=int(sys.argv[1]);r=fit('FullSplineWideLR2e4',62,seed,2e-4,12)
