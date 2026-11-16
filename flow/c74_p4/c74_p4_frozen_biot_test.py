#!/usr/bin/env python3
"""C74-P4 completely frozen independent original C32 Biot PDE test.

Test seed 742480 (not used in C64-C74). 2048 new prior draws and 90 noisy
pressure/displacement measurements, same alpha=.7, pulse and sigma as C73.
Test used ONCE after selecting modes and freezing three-seed checkpoints.
Proper conditional NLL excluding identical 512D Gaussian constant; lower better.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import time,json
from pathlib import Path
import numpy as np,pandas as pd,torch
import c74_p4_spline_flow as P4
P=P4.P
C=P4.DG.CF.JF.C;B=P4.DG.CF.JF.B;D=P4.DG.D
N=2048;SEED=742480
MODES=['C73_HMBlockSVDWhiten','BalancedAffine','FisherSpline','HMInfoSpline','RandomSpline']
SEEDS=[641133,641134,641135]
torch.set_num_threads(1)
fresh=P/'frozen_independent2048.npz'
if not fresh.exists():
    rg=np.random.default_rng(SEED);z=[];y=[];start=time.time()
    for i in range(N):
        w=rg.standard_normal(512)
        mat=D['base']+B.TM@w
        outcome=C.forward(mat[:256],mat[256:],.7,load='pulse')[0]
        assert np.isfinite(outcome).all()
        obs=outcome+D['sigma']*rg.standard_normal(90)
        z.append(w.astype('float32'));y.append(obs.astype('float32'))
        if (i+1)%256==0:print('FRESH ORIGINAL BIOT FORWARD',i+1,'/',N,'elapsed',round(time.time()-start,2),flush=True)
    np.savez_compressed(fresh,z=np.stack(z),y=np.stack(y),seed=SEED,
      protocol='Original C32 M1-linked Biot, alpha0.7 pulse, 512 latent fields, 42+48 measurements, independent C74-P4 cohort')
    (P/'frozen_physics_audit.json').write_text(json.dumps({'seed':SEED,'new_independent_truths':N,
        'original_biot_forward_calls':N,'elapsed_sec':time.time()-start,
        'physics':'Original C32 M1-linked, alpha=.7, pulse','network_selection_on_this_test':False},indent=2))
with np.load(fresh) as arr:z=arr['z'];y=arr['y']
mu,ctx=P4.DG.CF.feat(y)
X=torch.tensor(z)@torch.tensor(P4.W,dtype=torch.float32)
M=torch.tensor(mu)@torch.tensor(P4.W,dtype=torch.float32)
CC=torch.tensor(ctx)@torch.tensor(P4.DG.TRANSFORMS['HMBlockSVDWhiten'],dtype=torch.float32)
A=torch.tensor(P4.DG.A,dtype=torch.float32)@torch.tensor(P4.W,dtype=torch.float32)
X0=torch.tensor(z)@torch.tensor(P4.DG.W,dtype=torch.float32)
M0=torch.tensor(mu)@torch.tensor(P4.DG.W,dtype=torch.float32)
A0=torch.tensor(P4.DG.A,dtype=torch.float32)@torch.tensor(P4.DG.W,dtype=torch.float32)
raw={'case':np.arange(N)}
with torch.no_grad():
    v=X-M
    raw['Gaussian']=(.5*v.square().sum(1)+.5*(v@A.T).square().sum(1)-.5*P4.DG.CF.logdet).numpy()
for seed in SEEDS:
    for mode in MODES:
        if mode=='C73_HMBlockSVDWhiten':
            net=P4.DG.G.CondFlow(90)
            ck=P4.C73/'dual_checkpoints'/f'HMBlockSVDWhiten_seed{seed}.pt'
            Z0,Mu0,Ar0=X0,M0,A0
        else:
            net=P4.build_net(mode,seed)
            ck=P/'checkpoints'/f'{mode}_seed{seed}.pt'
            Z0,Mu0,Ar0=X,M,A
        net.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu'))
        net.eval()
        # split into batches for stability and fair memory
        scores=[];inverse_err=[]
        with torch.no_grad():
            for i in range(0,N,128):
                xx=Z0[i:i+128];mm=Mu0[i:i+128];cc=CC[i:i+128]
                uu,ld=net.inverse(xx,cc,mm)
                diff=uu-mm
                val=.5*diff.square().sum(1)+.5*(diff@Ar0.T).square().sum(1)-ld-.5*P4.DG.CF.logdet
                scores.append(val.numpy())
                recon=net.forward(uu,cc,mm)
                inverse_err.append(float((recon-xx).abs().max()))
        vals=np.concatenate(scores)
        assert np.isfinite(vals).all() and max(inverse_err)<2e-5,(mode,seed,max(inverse_err))
        raw[f'{mode}_{seed}']=vals
        print('TEST',mode,seed,'NLL',round(float(vals.mean()),6),'inv_err',max(inverse_err),flush=True)
raw=pd.DataFrame(raw)
raw.to_csv(P/'frozen2048_all_seeds.csv',index=False)
means=pd.DataFrame({'case':np.arange(N),'Gaussian':raw['Gaussian']})
for mode in MODES:
    means[mode]=raw[[f'{mode}_{seed}' for seed in SEEDS]].mean(axis=1)
means.to_csv(P/'frozen2048_mean_seeds.csv',index=False)
# Predefine 4 paired primary contrasts, all with strong physics controls.
contrasts=[('FisherSpline','C73_HMBlockSVDWhiten'),
           ('FisherSpline','RandomSpline'),
           ('FisherSpline','BalancedAffine'),
           ('HMInfoSpline','FisherSpline'),
           ('HMInfoSpline','RandomSpline')]
rg=np.random.default_rng(742481);stats=[]
for candidate,ref in contrasts:
    d=(means[candidate]-means[ref]).to_numpy(dtype=np.float64)
    # 5000 case-level bootstrap and paired sign flip, preserving model seed averaging.
    boot=np.empty(5000);flips=np.empty(5000)
    for j in range(5000):
        boot[j]=d[rg.integers(0,N,size=N)].mean()
        flips[j]=(d*rg.choice([-1,1],size=N)).mean()
    ci=np.quantile(boot,[.025,.975])
    p=(1+np.sum(abs(flips)>=abs(d.mean())))/(5001)
    stats.append({'candidate':candidate,'reference':ref,'n_independent_cases':N,
        'mean_delta_nll':d.mean(),'ci95_low':ci[0],'ci95_high':ci[1],
        'win_frequency':float((d<0).mean()),'signflip_p':p,
        'candidate_mean_nll':float(means[candidate].mean()),'reference_mean_nll':float(means[ref].mean())})
order=np.argsort([s['signflip_p'] for s in stats]);prev=0
for index,idx in enumerate(order):
    prev=max(prev,(len(stats)-index)*stats[idx]['signflip_p'])
    stats[idx]['holm_p']=min(1,prev)
pd.DataFrame(stats).to_csv(P/'frozen2048_paired_statistics.csv',index=False)
reps=[]
for seed in SEEDS:
    row={'seed':seed}
    for mode in MODES:row[mode]=float(raw[f'{mode}_{seed}'].mean())
    row['Fisher_vs_C73']=row['FisherSpline']-row['C73_HMBlockSVDWhiten']
    row['Fisher_vs_random']=row['FisherSpline']-row['RandomSpline']
    reps.append(row)
pd.DataFrame(reps).to_csv(P/'frozen2048_seed_replicates.csv',index=False)
summary={'n_fresh_original_biot_problems':N,'test_seed':SEED,'model_seeds':SEEDS,
 'means':{name:float(means[name].mean()) for name in ['Gaussian']+MODES},
 'contrasts':stats,
 'purpose':'Proper held-out conditional log density, not exact nonlinear posterior accuracy',
 'original_C73_unchanged':True,'model_selected_on_test':False}
(P/'frozen2048_summary.json').write_text(json.dumps(summary,indent=2))
print('FROZEN C74-P4 RESULTS',json.dumps(summary,indent=2),flush=True)
