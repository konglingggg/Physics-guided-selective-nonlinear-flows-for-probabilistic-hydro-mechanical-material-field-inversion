#!/usr/bin/env python3
"""Fresh confirmatory Biot synthetic holdout after validation selection (seed731960, 2048 cases).
Model checkpoints frozen before this dataset is generated. All 512D truths / all 90 observations.
Primary comparisons: HMBlockSVDWhiten vs Fisher, JointSVDWhiten,
RandomBlockWhiten, SensorDiagNorm; same architecture/parameter count.
Three NN initializations average per physical case; paired bootstrap and sign-flip over truths.
"""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,time,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import c73_dual_geometry_flow as DG
C=DG.CF.JF.C;B=DG.CF.JF.B;D=DG.D;G=DG.G
N=2048;SEED=731960
torch.set_num_threads(1)
fresh=ROOT/'dual_independent2048.npz'
if not fresh.exists():
    rng=np.random.default_rng(SEED);truth=[];obs=[];start=time.time()
    for i in range(N):
        z=rng.standard_normal(512)
        mat=D['base']+B.TM@z
        y=C.forward(mat[:256],mat[256:],.7,load='pulse')[0] + D['sigma']*rng.standard_normal(90)
        truth.append(z.astype('float32'));obs.append(y.astype('float32'))
        if (i+1)%256==0:print('ORIGINAL BIOT new dual cohort',i+1,'/',N,'time',round(time.time()-start,2),flush=True)
    np.savez_compressed(fresh,z=np.stack(truth),y=np.stack(obs),seed=SEED,sigma=D['sigma'],
        solver='Original C32 M1-linked Biot alpha0.7 pulse; 512 spatial material coords and 90 measurements')
    (ROOT/'dual_forward_audit.json').write_text(json.dumps({'new_independent_truths':N,'seed':SEED,'forward_solver':'C32 original','physics_runs':N,'seconds':time.time()-start},indent=2))
with np.load(fresh) as dat:z=dat['z'];y=dat['y']
mu,ctx=DG.CF.feat(y)
W=torch.tensor(DG.W,dtype=torch.float32)
ZZ=torch.tensor(z,dtype=torch.float32)@W
MU=torch.tensor(mu,dtype=torch.float32)@W
AR=torch.tensor(DG.A,dtype=torch.float32)@W
CTX=torch.tensor(ctx,dtype=torch.float32)
with torch.no_grad():
    diff=ZZ-MU
    gnll=.5*diff.square().sum(1)+.5*(diff@AR.T).square().sum(1)-.5*DG.CF.logdet
out={'case':np.arange(N),'Gaussian':gnll.numpy()}
# Frozen original 512D Fisher baseline from C73 A; compared with same architecture.
for seed in [641133,641134,641135]:
    fisher=G.CondFlow(90)
    fisher.load_state_dict(torch.load(ROOT/'checkpoints'/f'Fisher_seed{seed}.pt',map_location='cpu',weights_only=True))
    fisher.eval()
    with torch.no_grad():
        u,ld=fisher.inverse(ZZ,CTX,MU)
        d=u-MU
        score=.5*d.square().sum(1)+.5*(d@AR.T).square().sum(1)-ld-.5*DG.CF.logdet
        out[f'Fisher_{seed}']=score.numpy()
    for mode in ['HMBlockSVDWhiten','JointSVDWhiten','RandomBlockWhiten','SensorDiagNorm']:
        net=G.CondFlow(90)
        net.load_state_dict(torch.load(ROOT/'dual_checkpoints'/f'{mode}_seed{seed}.pt',map_location='cpu',weights_only=True))
        net.eval()
        c=CTX@torch.tensor(DG.TRANSFORMS[mode],dtype=torch.float32)
        with torch.no_grad():
            u,ld=net.inverse(ZZ,c,MU)
            diff=u-MU
            score=.5*diff.square().sum(1)+.5*(diff@AR.T).square().sum(1)-ld-.5*DG.CF.logdet
            rec=net.forward(u,c,MU)
            err=float((rec-ZZ).abs().max())
            assert err<2.e-5 and torch.isfinite(score).all(),(mode,seed,err)
            out[f'{mode}_{seed}']=score.numpy()
        print('FROZEN MODEL',mode,seed,'NLL',float(score.mean()),'inverse_err',err,flush=True)
raw=pd.DataFrame(out);raw.to_csv(ROOT/'dual_fresh2048_all_seeds.csv',index=False)
models=['Fisher','HMBlockSVDWhiten','JointSVDWhiten','RandomBlockWhiten','SensorDiagNorm']
for name in models:
    raw[name]=raw[[f'{name}_{s}' for s in [641133,641134,641135]]].mean(1)
raw[['case','Gaussian',*models]].to_csv(ROOT/'dual_fresh2048_mean_seeds.csv',index=False)
rg=np.random.default_rng(731961)
refs=['Fisher','JointSVDWhiten','RandomBlockWhiten','SensorDiagNorm']
stats=[]
for ref in refs:
    delta=(raw['HMBlockSVDWhiten']-raw[ref]).to_numpy().astype(float)
    samp=rg.integers(0,N,(6000,N))
    b=delta[samp].mean(1)
    ci=np.quantile(b,[.025,.975])
    flip=rg.choice([-1,1],size=(6000,N))
    p=(1+(np.abs((flip*delta).mean(1))>=abs(delta.mean())).sum())/6001
    stats.append({'contrast':'HMBlockSVDWhiten - '+ref,'mean_delta_nll':float(delta.mean()),
        'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),
        'improved_truths':int((delta<0).sum()),'n_truths':N,'signflip_p':float(p),
        'control_nll':float(raw[ref].mean()),'candidate_nll':float(raw['HMBlockSVDWhiten'].mean())})
idx=np.argsort([r['signflip_p'] for r in stats]);greatest=0
for rank,i in enumerate(idx):
    greatest=max(greatest,(len(idx)-rank)*stats[i]['signflip_p'])
    stats[i]['holm_p']=float(min(1,greatest))
pd.DataFrame(stats).to_csv(ROOT/'dual_fresh2048_primary_statistics.csv',index=False)
reps=[]
for seed in [641133,641134,641135]:
    vals={name:float(raw[f'{name}_{seed}'].mean()) for name in models}
    reps.append({'seed':seed,**vals,'HM_minus_Fisher':vals['HMBlockSVDWhiten']-vals['Fisher'],
      'HM_minus_RandomBlock':vals['HMBlockSVDWhiten']-vals['RandomBlockWhiten']})
pd.DataFrame(reps).to_csv(ROOT/'dual_fresh2048_seed_stability.csv',index=False)
summary={'new_seed':SEED,'fresh_original_biot_forward_cases':N,
 'model_train_true_pairs':1600,'model_validation_true_pairs':240,'three_seed_training_repeats':3,
 'trainable_parameters_each_model':302848,'means':{name:float(raw[name].mean()) for name in ['Gaussian']+models},
 'paired_comparisons':stats,
 'scope':'Conditioned predictive heldout material log density within fixed original C32 alpha0.7 pulse regime. Not exact nonlinear posterior accuracy or method novelty.'}
(ROOT/'dual_fresh2048_summary.json').write_text(json.dumps(summary,indent=2))
print('CONFIRMATORY RESULTS',json.dumps(summary,indent=2),flush=True)
