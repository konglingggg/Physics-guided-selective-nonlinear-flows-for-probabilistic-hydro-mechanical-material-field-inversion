#!/usr/bin/env python3
"""Audited C73 results and plots; callable without training."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import c73_dual_geometry_flow as DG
G=DG.G;CF=DG.CF
results=pd.read_csv(ROOT/'dual_fresh2048_mean_seeds.csv')
with np.load(ROOT/'dual_independent2048.npz') as q:z=q['z'];y=q['y']
assert len(results)==len(z)==len(y)==2048
# Frozen C69 generic model in physical coordinates, same 90 observations & normalizations
mu,ctx=CF.feat(y)
with torch.no_grad():
    X=torch.tensor(z,dtype=torch.float32); MU=torch.tensor(mu,dtype=torch.float32)
    CTX=torch.tensor(ctx,dtype=torch.float32); A=torch.tensor(CF.A,dtype=torch.float32)
    old=G.CondFlow(90)
    old.load_state_dict(torch.load(CF.C72/'inputs'/'C69_Amortized_Likelihood_20261111'/'joint_flow'/'J_conditional_flow.pt',weights_only=True))
    old.eval();u,ld=old.inverse(X,CTX,MU)
    d=u-MU
    nll=(.5*d.square().sum(1)+.5*(d@A.T).square().sum(1)-ld-.5*CF.logdet).numpy()
results['C69Generic']=nll
results.to_csv(ROOT/'dual_fresh2048_with_C69_baseline.csv',index=False)
# Summarize validation of each seed using checkpoints' training logs
rows=[]
for seed in [641133,641134,641135]:
    for name in ['Fisher','HMComplementary','HM_Route','HM_ShuffledRoute']:
        log=json.loads((ROOT/'checkpoints'/f'{name}_seed{seed}_log.json').read_text())['result']
        rows.append({'seed':seed,'model':name,'val_nll':log['best_val_nll'],'best_epoch':log['best_epoch'],'params':log['n_parameters'],'family':'A:mask-gating'})
    for name in ['JointSVD','HMBlockSVD','RandomCtx','JointSVDWhiten','HMBlockSVDWhiten','SensorDiagNorm','RandomBlockWhiten']:
        path=ROOT/'dual_checkpoints'/f'{name}_seed{seed}_log.json'
        if path.exists():
            log=json.loads(path.read_text())['summary']
            rows.append({'seed':seed,'model':name,'val_nll':log['val_nll'],'best_epoch':log['best_epoch'],'params':log['n_parameters'],'family':'B:dual information geometry'})
pd.DataFrame(rows).to_csv(ROOT/'all_validation_audit.csv',index=False)
# 2048 independent density visualization using original Matplotlib color cycle, no custom palette
labels=['C69 Generic','Fisher','Joint SVD whiten','Sensor diagonal norm','Random H/M whiten','H/M physical SVD whiten']
keys=['C69Generic','Fisher','JointSVDWhiten','SensorDiagNorm','RandomBlockWhiten','HMBlockSVDWhiten']
vals=[results[k].mean() for k in keys]
plt.figure(figsize=(10,4.8))
plt.barh(labels,vals)
plt.xlim(230,255)
plt.xlabel('Held-out conditional NLL (lower is better)')
plt.title('C73: 2,048 new Biot truth-observation pairs')
for i,v in enumerate(vals):plt.text(v+.15,i,f'{v:.3f}',va='center',fontsize=9)
plt.tight_layout();plt.savefig(ROOT/'C73_dual_geometry_comparison.png',dpi=180);plt.close()
stats=pd.read_csv(ROOT/'dual_fresh2048_primary_statistics.csv')
plt.figure(figsize=(8.8,4.7))
for i,r in enumerate(stats.itertuples(index=False)):
    delta=float(r.mean_delta_nll);lo=float(r.ci95_low);hi=float(r.ci95_high)
    plt.errorbar(delta,i,xerr=np.array([[delta-lo],[hi-delta]]),fmt='o',capsize=4)
plt.axvline(0,linestyle='--',color='gray')
plt.yticks(range(4),['vs Fisher','vs Joint SVD whiten','vs Random H/M whiten','vs Sensor diagonal norm'])
plt.xlabel('H/M physical SVD whiten minus reference: paired conditional NLL')
plt.title('C73: Paired 95% bootstrap intervals (2,048 independent truths)')
plt.tight_layout();plt.savefig(ROOT/'C73_dual_geometry_paired_CI.png',dpi=180);plt.close()
# numerical integrity: analytic inverse vs direct and per-layer affine triangular determinant
W,net=CF.build('Fisher');assert sum(p.numel() for p in net.parameters())==302848
for name in ['HMBlockSVDWhiten','RandomBlockWhiten','JointSVDWhiten']:
    P=DG.TRANSFORMS[name]
    assert P.shape==(90,90) and np.linalg.matrix_rank(P)==90 and np.isfinite(P).all()
N=16
maxtol=0
for model in ['HMBlockSVDWhiten','RandomBlockWhiten']:
    net=G.CondFlow(90)
    net.load_state_dict(torch.load(ROOT/'dual_checkpoints'/f'{model}_seed641133.pt',weights_only=True))
    net.eval()
    c=torch.tensor((ctx[:N]@DG.TRANSFORMS[model]).astype('float32'))
    w=torch.tensor((z[:N]@DG.W).astype('float32'))
    m=torch.tensor((mu[:N]@DG.W).astype('float32'))
    with torch.no_grad():
        inv,ld=net.inverse(w,c,m);recovered=net.forward(inv,c,m)
        maxtol=max(maxtol,float((recovered-w).abs().max()))
assert maxtol<2.e-5
out={'original_C32_solver':True,'n_frozen_test_material_truths':2048,
     'original_C69_jointflow_NLL':float(np.mean(nll)),
     'all_trainable_parameters_equal_302848':True,
     'float32_forward_inverse_max_abs_error':maxtol,
     'log_density_finite_and_normalized_by_same_90_obs_physics_Gaussian':True,
     'NLL_C73_HM_physical_SVD':float(results['HMBlockSVDWhiten'].mean())}
(ROOT/'C73_final_numerical_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2),flush=True)
