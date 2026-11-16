#!/usr/bin/env python3
"""C74-P5: compare frozen Fisher-allocated partial-spline with full-coordinate spline strong controls.
Train (1600) and choose epoch on original development validation (240) ONLY.
Final frozen scores on separate 1024 freshly generated original-C32 Biot conditions.
Out-of-distribution sensitivity alpha=.4,1 compared to nominal alpha=.7, fixed pulse.
No architectural revisions after seeing frozen performance. Full Spline is established technology.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,argparse
from pathlib import Path
import numpy as np,torch,pandas as pd
from torch import nn
HERE=Path(__file__).resolve().parent
P=HERE.parent/'C74_P4_PhysicsAllocated_Spline_20261111'
sys.path.insert(0,str(P))
import c74_p4_spline_flow as P4
C73=P4.C73;D=P4.DG

class FullCoordinateSpline(nn.Module):
    def __init__(self,hidden=18):
        super().__init__()
        old=P4.N_SPLINE
        try:
            P4.N_SPLINE=256
            self.layers=nn.ModuleList([P4.PartialSplineCoupling(np.arange(256),swap=bool(i%2),hidden=hidden) for i in range(4)])
        finally:
            P4.N_SPLINE=old
    def forward_with_ld(self,u,c,mu):
        x=u-mu;ld=torch.zeros(len(u),dtype=u.dtype,device=u.device)
        for l in self.layers:
            x,v=l.forward(x,c);ld=ld+v
        return x+mu,ld
    def forward(self,u,c,mu):return self.forward_with_ld(u,c,mu)[0]
    def inverse(self,z,c,mu):
        x=z-mu;ld=torch.zeros(len(z),dtype=z.dtype,device=z.device)
        for l in reversed(self.layers):
            x,v=l.inverse(x,c);ld=ld+v
        return x+mu,ld

def build(mode,seed):
    if mode=='FullSplineMatched':return FullCoordinateSpline(18)
    if mode=='FullSplineWide':return FullCoordinateSpline(62)
    if mode=='FullSplineWideLR2e4':return FullCoordinateSpline(62)
    if mode=='WeakSpline':return P4.PartialSplineFlow('WeakSpline',seed)
    if mode=='FisherSpline':return P4.PartialSplineFlow('FisherSpline',seed)
    raise KeyError(mode)

def nll(net,z,mu,c,AR,logdet):
    u,ld=net.inverse(z,c,mu)
    d=u-mu
    return .5*d.square().sum(1)+.5*(d@AR.T).square().sum(1)-ld-.5*logdet

def train_one(mode,seed,epochs=8):
    torch.set_num_threads(1);torch.manual_seed(seed);r=np.random.default_rng(seed)
    model=build(mode,seed);nparams=sum(p.numel() for p in model.parameters())
    optim=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=5e-5)
    hist=[];best=1e6;bestep=0
    z=P4.DATA_Z;mu=P4.DATA_MU;c=P4.DATA_CTX;AR=P4.AR
    ck=HERE/'checkpoints'/f'{mode}_seed{seed}.pt';ck.parent.mkdir(exist_ok=True)
    start=time.time()
    for ep in range(1,epochs+1):
        model.train()
        for ids in np.array_split(r.permutation(1600),20):
            optim.zero_grad()
            x=z[ids];m=mu[ids];ct=c[ids]
            u,ld=model.inverse(x,ct,m)
            d=u-m
            l=.5*d.square().sum(1)+.5*(d@AR.T).square().sum(1)-ld-.5*P4.DG.CF.logdet
            reg=.3*(u-x).square().mean()
            loss=l.mean()+reg
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training {mode}/{seed}/{ep}')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.5);optim.step()
        model.eval()
        with torch.no_grad():
            val=float(nll(model,z[1600:1840],mu[1600:1840],c[1600:1840],AR,P4.DG.CF.logdet).mean())
        hist.append({'epoch':ep,'val_nll':val})
        if val<best-1e-5:
            best=val;bestep=ep;torch.save(model.state_dict(),ck)
        print('EPOCH',mode,seed,ep,round(val,5),'BEST',bestep,round(best,5),flush=True)
    model.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu'));model.eval()
    with torch.no_grad():
        subset=z[1600:1603];cc=c[1600:1603];mm=mu[1600:1603]
        u,ldi=model.inverse(subset,cc,mm);re,ldf=model.forward_with_ld(u,cc,mm)
    row={'mode':mode,'seed':seed,'parameters':nparams,'best_epoch':bestep,'val_nll':best,'seconds':round(time.time()-start,3),
         'inverse_error':float((subset-re).abs().max()),'logdet_consistency_error':float((ldi+ldf).abs().max()),'checkpoint':str(ck)}
    (HERE/'checkpoints'/f'{mode}_seed{seed}_train.json').write_text(json.dumps({'summary':row,'history':hist},indent=2))
    print('TRAIN',json.dumps(row),flush=True)
    return row

def draw_fresh(seed,n,alpha):
    C=D.CF.JF.C;B=D.CF.JF.B;dat=D.D
    rng=np.random.default_rng(seed)
    zz=np.zeros((n,512),dtype=np.float32);yy=np.zeros((n,90),dtype=np.float32)
    for i in range(n):
        z=rng.standard_normal(512)
        m=dat['base']+B.TM@z
        f=C.forward(m[:256],m[256:],alpha,load='pulse')[0]
        if not np.all(np.isfinite(f)):raise RuntimeError(f'PDE fail at {alpha}/{i}')
        yy[i]=f+dat['sigma']*rng.standard_normal(90)
        zz[i]=z
    return zz,yy

def load_net(mode,seed):
    if mode=='C73':
        net=D.G.CondFlow(90);ck=C73/'dual_checkpoints'/f'HMBlockSVDWhiten_seed{seed}.pt'
    elif mode in ('FisherSpline','HMInfoSpline','RandomSpline'):
        net=P4.build_net(mode,seed);ck=P/'checkpoints'/f'{mode}_seed{seed}.pt'
    else:
        net=build(mode,seed);ck=HERE/'checkpoints'/f'{mode}_seed{seed}.pt'
    net.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu'));net.eval()
    return net

@torch.no_grad()
def score(net,mode,xx,mm,cc,a,logdet):
    arr=[]
    for j in range(0,len(xx),128):
        vals=nll(net,xx[j:j+128],mm[j:j+128],cc[j:j+128],a,logdet)
        if not torch.isfinite(vals).all():raise RuntimeError('nonfinite NLL '+mode)
        arr.append(vals.numpy())
    return np.concatenate(arr)

def eval_cohort(seed,n,alpha,cohort_name,methods):
    path=HERE/f'cohort_{cohort_name}.npz'
    if not path.exists():
        t0=time.time();z,y=draw_fresh(seed,n,alpha)
        np.savez_compressed(path,z=z,y=y,alpha=np.array(alpha),seed=np.array(seed))
        print('ORIGINAL BIOT NEW COHORT',cohort_name,n,alpha,'seconds',time.time()-t0,flush=True)
    else:
        with np.load(path) as ff:z=ff['z'];y=ff['y']
    mu,ctx=D.CF.feat(y)
    W=P4.W
    x=torch.from_numpy((z@W).astype('float32'))
    mm=torch.from_numpy((mu@W).astype('float32'))
    c=torch.from_numpy((ctx@D.TRANSFORMS['HMBlockSVDWhiten']).astype('float32'))
    a=torch.from_numpy((D.A@W).astype('float32'))
    w0=D.W
    x0=torch.from_numpy((z@w0).astype('float32'))
    m0=torch.from_numpy((mu@w0).astype('float32'))
    a0=torch.from_numpy((D.A@w0).astype('float32'))
    out={'case':np.arange(len(z))}
    with torch.no_grad():
        v=x-mm;out['NominalTangentGaussian']=(.5*v.square().sum(1)+.5*(v@a.T).square().sum(1)-.5*D.CF.logdet).numpy()
    for mode in methods:
        for seed_model in (641133,641134,641135):
            net=load_net(mode,seed_model)
            xx,uu,aa=(x0,m0,a0) if mode=='C73' else (x,mm,a)
            vals=score(net,mode,xx,uu,c,aa,D.CF.logdet)
            out[f'{mode}_{seed_model}']=vals
            print('SCORE',cohort_name,mode,seed_model,round(float(vals.mean()),5),flush=True)
    df=pd.DataFrame(out);df.to_csv(HERE/f'{cohort_name}_raw_scores.csv',index=False)
    mean=pd.DataFrame({'case':df['case'],'NominalTangentGaussian':df['NominalTangentGaussian']})
    for mode in methods:
        cols=[c for c in df.columns if c.startswith(f'{mode}_')]
        mean[mode]=df[cols].mean(axis=1)
    mean.to_csv(HERE/f'{cohort_name}_mean_scores.csv',index=False)
    return mean

def paired(df,contrasts,seed):
    rng=np.random.default_rng(seed);out=[];n=len(df)
    for x,y in contrasts:
        delta=(df[x]-df[y]).to_numpy(np.float64)
        boot=np.empty(3000)
        for i in range(len(boot)):boot[i]=np.mean(delta[rng.integers(n,size=n)])
        out.append({'candidate':x,'reference':y,'n_cases':n,'delta_nll':delta.mean(),'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),
                    'wins':(delta<0).sum(),'wins_fraction':(delta<0).mean()})
    return pd.DataFrame(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--test',action='store_true');ap.add_argument('--epochs',type=int,default=8)
    opt=ap.parse_args();torch.set_num_threads(1)
    if opt.train:
        rows=[]
        for name in ['WeakSpline','FullSplineMatched','FullSplineWide']:
            for seed in (641133,641134,641135):
                rows.append(train_one(name,seed,opt.epochs))
        pd.DataFrame(rows).to_csv(HERE/'development_new_baselines.csv',index=False)
    if opt.test:
        methods=['C73','FisherSpline','HMInfoSpline','RandomSpline','WeakSpline','FullSplineMatched','FullSplineWide','FullSplineWideLR2e4']
        conditions=[(743521,1024,.7,'fresh1024_nominal'),(744521,1024,.7,'fresh1024_tuned_confirm'),(743522,256,.4,'shift256_alpha04'),(743523,256,1.0,'shift256_alpha10')]
        allrows=[]
        for seed,n,alpha,name in conditions:
            df=eval_cohort(seed,n,alpha,name,methods)
            contrasts=[('FisherSpline','C73'),('FisherSpline','FullSplineMatched'),('FisherSpline','FullSplineWide'),
                ('FisherSpline','WeakSpline'),('FisherSpline','RandomSpline'),('FisherSpline','FullSplineWideLR2e4'),('HMInfoSpline','FisherSpline')]
            stat=paired(df,contrasts,seed+99);stat.insert(0,'cohort',name);stat.insert(1,'alpha',alpha)
            stat.to_csv(HERE/f'{name}_paired.csv',index=False)
            allrows+=stat.to_dict('records')
        pd.DataFrame(allrows).to_csv(HERE/'all_conditions_paired.csv',index=False)
        print('PAIRED STATS',pd.DataFrame(allrows).to_string(index=False),flush=True)
