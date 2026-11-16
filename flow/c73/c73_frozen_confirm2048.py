#!/usr/bin/env python3
"""C73 larger independent Biot confirmation. Uses seed731255 not seen in C64/C69/C72.
The 4 matched model variants x3 seeds are frozen before this module is run.
Primary contrasts: HMRouted vs Fisher; HMRouted vs same permutation without dynamic physics;
HMRouted vs physically shuffled dynamic gate. Bootstrap resamples independent *material truths*,
not NN seeds. Holm correction on three primary two-sided paired sign-flip tests.
"""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import c73_route_flow as CF
JF=CF.JF;G=CF.G; D=CF.D
C=JF.C;B=JF.B
torch.set_num_threads(1)
N=2048;SEED=731900
out=ROOT/'independent2048.npz'
if not out.exists():
    rng=np.random.default_rng(SEED)
    zs=[];ys=[];start=time.time();failures=[]
    for i in range(N):
        z=rng.standard_normal(512)
        mat=D['base']+B.TM@z
        try:
            yclean=C.forward(mat[:256],mat[256:],.7,load='pulse')[0]
            y=yclean+D['sigma']*rng.standard_normal(90)
            zs.append(z.astype('float32'));ys.append(y.astype('float32'))
        except Exception as exc:
            failures.append({'case':i,'error':repr(exc)})
            raise
        if (i+1)%32==0:
            print('INDEPENDENT ORIGINAL BIOT',i+1,'/',N,'elapsed',round(time.time()-start,2),'seconds',flush=True)
    np.savez_compressed(out,z=np.array(zs),y=np.array(ys),seed=SEED,
        description='Original C32 M1-linked Biot full 512D logk logE, H/M90, alpha0.7 pulse',sigma=D['sigma'])
    (ROOT/'fresh_generator_audit.json').write_text(json.dumps({'n':N,'seed':SEED,'physics':'C32 M1-linked Biot','alpha':.7,'load':'pulse','error_count':len(failures),'elapsed_sec':time.time()-start},indent=2))
with np.load(out) as q: z=q['z'];y=q['y']
mu,ctx=CF.feat(y)
X=torch.tensor(z,dtype=torch.float32);M=torch.tensor(mu,dtype=torch.float32);CTX=torch.tensor(ctx,dtype=torch.float32)
rawA=torch.tensor(CF.A,dtype=torch.float32)
with torch.no_grad():
    d=X-M
    gauss=(.5*d.square().sum(1)+.5*(d@rawA.T).square().sum(1)-.5*CF.logdet).numpy()
result={'Gaussian':gauss}
# C69 original independent training / same 90 obs
orig=G.CondFlow(90)
orig.load_state_dict(torch.load(CF.C72/'inputs'/'C69_Amortized_Likelihood_20261111'/'joint_flow'/'J_conditional_flow.pt',weights_only=True,map_location='cpu'))
orig.eval()
with torch.no_grad():
    u,ld=orig.inverse(X,CTX,M)
    d=u-M
    result['C69_JointFlow']=(.5*d.square().sum(1)+.5*(d@rawA.T).square().sum(1)-ld-.5*CF.logdet).numpy()
for seed in [641133,641134,641135]:
    for mode in ['Fisher','HMComplementary','HM_Route','HM_ShuffledRoute']:
        W,net=CF.build(mode)
        net.load_state_dict(torch.load(ROOT/'checkpoints'/f'{mode}_seed{seed}.pt',weights_only=True,map_location='cpu'))
        net.eval()
        WW=torch.tensor(W.astype('float32'))
        xx=X@WW; mm=M@WW; ar=rawA@WW
        with torch.no_grad():
            u,ld=net.inverse(xx,CTX,mm)
            delta=u-mm
            score=(.5*delta.square().sum(1)+.5*(delta@ar.T).square().sum(1)-ld-.5*CF.logdet).numpy()
            # Inverse numerical check on new data, plus finite density.
            rec=net.forward(u,CTX,mm)
            error=float((rec-xx).abs().max())
            assert np.isfinite(score).all() and error<2e-5, (mode,seed,error)
        result[f'{mode}_{seed}']=score
        print('FROZEN',mode,seed,'mean',float(score.mean()),'max_inverse_error',error,flush=True)
frame=pd.DataFrame({'case':np.arange(N),**result})
frame.to_csv(ROOT/'fresh2048_all_models.csv',index=False)
# Independent cases first, averaged over three training seeds to prevent pseudo-replicating test truths.
for mode in ['Fisher','HMComplementary','HM_Route','HM_ShuffledRoute']:
    frame[mode]=frame[[f'{mode}_{s}' for s in [641133,641134,641135]]].mean(axis=1)
frame[['case','Gaussian','C69_JointFlow','Fisher','HMComplementary','HM_Route','HM_ShuffledRoute']].to_csv(ROOT/'fresh2048_mean_across_seeds.csv',index=False)
rng=np.random.default_rng(731901)
primary=[('HM_Route','Fisher'),('HM_Route','HMComplementary'),('HM_Route','HM_ShuffledRoute')]
rows=[]
for model,reference in primary:
    delta=(frame[model]-frame[reference]).to_numpy().astype(float)
    idx=rng.integers(0,N,size=(6000,N))
    boot=delta[idx].mean(axis=1)
    ci=np.quantile(boot,[.025,.975])
    signs=rng.choice([-1,1],size=(6000,N),replace=True)
    observed=abs(delta.mean());p=float((1+(np.abs((delta*signs).mean(1))>=observed).sum())/(len(signs)+1))
    rows.append({'candidate':model,'reference':reference,'n_independent_Biot_truths':N,
                 'mean_NLL_delta':float(delta.mean()),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),
                 'better_truths':int((delta<0).sum()),'mean_base':float(frame[reference].mean()),
                 'mean_candidate':float(frame[model].mean()),'signflip_p':p})
# Holm
order=np.argsort([r['signflip_p'] for r in rows]);k=len(rows);adj=np.zeros(k)
maxv=0
for j,pos in enumerate(order):
    maxv=max(maxv,(k-j)*rows[pos]['signflip_p']);adj[pos]=min(1,maxv)
for i,v in enumerate(adj):rows[i]['holm_p']=float(v)
pd.DataFrame(rows).to_csv(ROOT/'fresh2048_primary_paired_tests.csv',index=False)
seedrows=[]
for seed in [641133,641134,641135]:
    row={'seed':seed}
    for mode in ['Fisher','HMComplementary','HM_Route','HM_ShuffledRoute']:
        row[f'{mode}_NLL']=float(frame[f'{mode}_{seed}'].mean())
    for ref in ['Fisher','HMComplementary','HM_ShuffledRoute']:
        dd=(frame[f'HM_Route_{seed}']-frame[f'{ref}_{seed}']).to_numpy()
        row[f'HM_Route_minus_{ref}']=float(dd.mean())
    seedrows.append(row)
pd.DataFrame(seedrows).to_csv(ROOT/'fresh2048_seed_replicates.csv',index=False)
meta={'fresh_seed':SEED,'n_fresh_truths':N,'n_train_initializations':3,
      'independent_test_selected_after_validation':True,
      'metric':'proper conditional NLL up to universal Gaussian constant; smaller better',
      'means':{key:float(frame[key].mean()) for key in ['Gaussian','C69_JointFlow','Fisher','HMComplementary','HM_Route','HM_ShuffledRoute']},
      'primary_paired_tests':rows,
      'disclaimer':'Synthetic original C32 Biot independent field/noise cohort at same physics regime. Conditional heldout log-score, not exact nonlinear posterior calibration, and does not establish method originality.'}
(ROOT/'fresh2048_summary.json').write_text(json.dumps(meta,indent=2))
print('FINAL FROZEN SUMMARY',json.dumps(meta,indent=2),flush=True)
