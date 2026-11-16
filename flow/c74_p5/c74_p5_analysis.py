#!/usr/bin/env python3
"""Recompute case-level paired confidence intervals and Holm-adjusted sign-flip tests.
Single original Biot true material field is the independent sampling unit.
Three network seeds are averaged within case before inferential testing.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
rng=np.random.default_rng(744534)
source=pd.read_csv(P/'fresh1024_tuned_confirm_mean_scores.csv')
comp=[('FisherSpline','C73'),('FisherSpline','FullSplineMatched'),('FisherSpline','FullSplineWideLR2e4'),('FisherSpline','WeakSpline'),('FisherSpline','RandomSpline'),('HMInfoSpline','FisherSpline')]
rows=[]
for method,ref in comp:
    d=(source[method]-source[ref]).to_numpy(float)
    b=np.mean(d[rng.integers(len(d),size=(5000,len(d)))],axis=1)
    flips=np.mean(d*rng.choice([-1,1],size=(5000,len(d))),axis=1)
    p=(1+(np.abs(flips)>=abs(d.mean())).sum())/5001
    rows.append({'method':method,'reference':ref,'n_cases':len(d),'mean_delta_NLL':float(d.mean()),
         'bootstrap95_low':float(np.quantile(b,.025)),'bootstrap95_high':float(np.quantile(b,.975)),
         'signflip_two_sided_p':float(p),'win_fraction':float((d<0).mean())})
ix=np.argsort([r['signflip_two_sided_p'] for r in rows]);step=0
for j,k in enumerate(ix):
    step=max(step,(len(rows)-j)*rows[k]['signflip_two_sided_p']);rows[k]['Holm_adjusted_p']=min(1.,step)
pd.DataFrame(rows).to_csv(P/'primary_confirmation_full_statistics.csv',index=False)
training=[]
for f in sorted((P/'checkpoints').glob('*_train.json')):
    info=json.loads(f.read_text())['summary'];training.append(info)
pd.DataFrame(training).to_csv(P/'all_training_audit.csv',index=False)
methods=['C73','FullSplineMatched','FullSplineWideLR2e4','RandomSpline','WeakSpline','FisherSpline','HMInfoSpline']
means=source[methods].mean().sort_values()
fig,ax=plt.subplots(figsize=(9,4.5))
ax.barh(list(reversed(means.index)),list(reversed(means.values)))
ax.set_xlabel('Conditional NLL (lower is better)')
ax.set_title('New 1024-case Biot cohort: frozen 512D conditional density')
ax.set_xlim(236,243)
for i,v in enumerate(list(reversed(means.values))):ax.text(v+.05,i,f'{v:.3f}',va='center',fontsize=9)
fig.tight_layout();fig.savefig(P/'C74_P5_full_spline_confirmation.png',dpi=160);plt.close(fig)
subset=pd.read_csv(P/'exact_posterior_pilot_cases.csv')
ess=subset.pivot(index='case',columns='method',values='ESS')
fig,ax=plt.subplots(figsize=(9,4.8))
x=np.arange(len(ess));col=['MAPGaussian','C73','FisherSpline','FullSplineMatched']
for i,key in enumerate(col):ax.bar(x+(i-1.5)*.18,ess[key],width=.18,label=key)
ax.set_xticks(x,[str(v) for v in ess.index]);ax.set_xlabel('Independent observed Biot material case')
ax.set_ylabel('Importance ESS / 96 exact PDE likelihood evaluations')
ax.set_title('Nonlinear posterior diagnostic: predictive NLL is not posterior ESS')
ax.legend(fontsize=8);fig.tight_layout();fig.savefig(P/'C74_P5_exact_posterior_ESS.png',dpi=160);plt.close(fig)
report={'fresh_confirm_1024':{'n_cases':len(source),'mean_nll':means.to_dict(),'primary_contrasts':rows},
        'exact_nonlinear_pilot':{'n_observations':int(subset['case'].nunique()),'n_samples_per_method_per_observation':96,
        'mean_ess':subset.groupby('method')['ESS'].mean().to_dict(),
        'mean_weighted_logE_RMSE':subset.groupby('method')['logE_weighted_RMSE'].mean().to_dict()},
        'ood_status':'alpha=0.4 and 1.0 use frozen nominal alpha=0.7 physics tangent, so absolute NLL is not calibrated. Interpret only exploratory sensitivity.'}
(P/'P5_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
print('ESS',report['exact_nonlinear_pilot']['mean_ess'])
