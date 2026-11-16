#!/usr/bin/env python3
"""C73: exact-density H/M information-routed conditional RealNVP at 512D.
Train/val: unchanged C64 original C32 Biot 1600/240. NO fresh test selection.
Compared routes: Fisher, rank-balanced Fisher, HM-complementary Fisher,
HM-complementary+physics gate, shuffled physics gate, Fisher+physics gate.
Every candidate has exactly the same 302,848 trainable parameters, four affine coupling layers,
prior, 90 observations, training epochs, optimiser and validation selection.
"""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
C72=(ROOT/'deps'/'C72_InformationGeometry_Pilot_20261111' if (ROOT/'deps'/'C72_InformationGeometry_Pilot_20261111').exists() else Path('/mnt/data/C72_InformationGeometry_Pilot_20261111'))
sys.path.insert(0,str(C72/'inputs'/'C69_Amortized_Likelihood_20261111'))
import c69_joint_flow as JF
G=JF.G
D=JF.data
torch.set_num_threads(1)
A,feat,logdet,_=JF.init_joint(D,'J')
mu,ctx=feat(D['y']);z=D['z']
phys=np.load(C72/'fisher_basis.npz')
V=phys['V'].astype('float64');ev=phys['eigenvalues']
AH=phys['AH'].astype('float64');AM=phys['AM'].astype('float64')
h=np.square(AH@V).sum(0); m=np.square(AM@V).sum(0)
# Deterministic physics routing 512->512 permutation: first 256 H-dominant and weak residual,
# remaining 256 M-dominant and residual. 90 high-information directions are not discarded.
hm_H=list(np.flatnonzero(h>m));hm_M=list(np.flatnonzero(h<=m))
# Assign negligible-information coordinates to fill each side (do not remove them).
strong=np.flatnonzero(ev>1.e-5)
Hstrong=[i for i in strong if h[i]>m[i]]; Mstrong=[i for i in strong if h[i]<=m[i]]
weak=[i for i in range(512) if i not in set(strong.tolist())]
hleft=Hstrong+weak[:256-len(Hstrong)]
mright=Mstrong+weak[256-len(Hstrong):]
assert len(hleft)==256 and len(mright)==256
perm_HM=np.array(hleft+mright,dtype=int)
assert len(set(perm_HM))==512
# Rank-balanced control: alternate high-sensitivity axes across halves, weak split by index.
info=list(strong); weak_rank=[i for i in range(512) if i not in set(info)]
left=info[::2];right=info[1::2]
need=256-len(left)
perm_rank=np.array(left+weak_rank[:need]+right+weak_rank[need:],dtype=int)
assert len(perm_rank)==512 and len(set(perm_rank))==512

class RoutedLayer(G.Coupling):
    def __init__(self,nobs=90,swap=False,gate_strength=.5):
        super().__init__(nobs,hidden=80,swap=swap)
        self.gate_strength=gate_strength
    def forward(self,x,c,gate=None):
        a,b=self.split(x);s,t=self.params(a,c)
        if gate is not None:
            outgate=(gate[:,:256] if self.swap else gate[:,256:])
            s=s*outgate;t=t*outgate
        return self.merge(a,b*torch.exp(s)+t),s.sum(1)
    def inverse(self,x,c,gate=None):
        a,b=self.split(x);s,t=self.params(a,c)
        if gate is not None:
            outgate=(gate[:,:256] if self.swap else gate[:,256:])
            s=s*outgate;t=t*outgate
        return self.merge(a,(b-t)*torch.exp(-s)),-s.sum(1)
class HMRoutedFlow(nn.Module):
    def __init__(self, nobs, W, shuffled=False, strength=.5):
        super().__init__(); self.layers=nn.ModuleList([RoutedLayer(nobs,swap=bool(i%2)) for i in range(4)])
        # Fixed differentiable route: gradient contributions of H and M observation likelihood in Fisher coords.
        H=AH@W;M=AM@W
        inf=np.sum(H*H,axis=0)+np.sum(M*M,axis=0)
        scale=(inf/(1+inf))
        coeff_H=H/np.sqrt(1+inf)[None,:]
        coeff_M=M/np.sqrt(1+inf)[None,:]
        if shuffled:
            rg=np.random.default_rng(731311)
            pp=rg.permutation(512)
            coeff_H=coeff_H[:,pp];coeff_M=coeff_M[:,pp];scale=scale[pp]
        self.register_buffer('P_H',torch.tensor(coeff_H,dtype=torch.float32))
        self.register_buffer('P_M',torch.tensor(coeff_M,dtype=torch.float32))
        self.register_buffer('info',torch.tensor(scale,dtype=torch.float32))
        self.strength=strength
    def route(self,ctx):
        # Original context = scaled standardized observation residual r/10.
        # Clamped smooth, positive, data-dependent diagonal gate; independent of the material being transformed.
        uH=ctx[:,:42]@self.P_H
        uM=ctx[:,42:]@self.P_M
        return 1. + self.strength*self.info.unsqueeze(0)*torch.tanh((uH-uM)*2.)
    def inverse(self,z,ctx,mu):
        x=z-mu;ld=torch.zeros(len(z),device=z.device)
        gate=self.route(ctx)
        for layer in reversed(self.layers):
            x,l=layer.inverse(x,ctx,gate);ld+=l
        return x+mu,ld
    def forward(self,z0,ctx,mu):
        x=z0-mu;gate=self.route(ctx)
        for layer in self.layers:x,_=layer(x,ctx,gate)
        return x+mu

def build(mode):
    if mode in ('Fisher','Fisher_Route'):p=np.arange(512)
    elif mode=='RankBalanced':p=perm_rank
    else:p=perm_HM
    W=V[:,p]
    if mode in ('HM_Route','HM_ShuffledRoute','Fisher_Route'):
        net=HMRoutedFlow(90,W,shuffled=(mode=='HM_ShuffledRoute'))
    else:net=G.CondFlow(90)
    return W,net

def train(mode, seed=641133, epochs=32, batch=80):
    start=time.time()
    torch.manual_seed(seed);r=np.random.default_rng(seed)
    W,net=build(mode)
    Z=torch.from_numpy((z@W).astype('float32'))
    MU=torch.from_numpy((mu@W).astype('float32'))
    CT=torch.from_numpy(ctx.astype('float32'))
    AR=torch.from_numpy((A@W).astype('float32'))
    opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=5e-5)
    def evaluate(ids,reg=False):
        zz=Z[ids];mm=MU[ids];cc=CT[ids]
        u,ld=net.inverse(zz,cc,mm)
        d=u-mm
        nll=.5*d.square().sum(1)+.5*(d@AR.T).square().sum(1)-ld-.5*logdet
        return nll.mean()+(.3*(u-zz).square().mean() if reg else 0),nll
    best=1e12;bestep=0;records=[]
    output=ROOT/'checkpoints';output.mkdir(exist_ok=True)
    ck=output/f'{mode}_seed{seed}.pt'
    for ep in range(1,epochs+1):
        net.train();pp=r.permutation(1600)
        for i in range(0,1600,batch):
            ids=pp[i:i+batch];opt.zero_grad();loss,_=evaluate(ids,True)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.5);opt.step()
        net.eval()
        with torch.no_grad():vl,_=evaluate(np.arange(1600,1840))
        val=float(vl)
        records.append({'epoch':ep,'val_nll':val})
        if val<best-1e-5:
            best=val;bestep=ep;torch.save(net.state_dict(),ck)
    net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True));net.eval()
    # C64 reserved test is a development/selection diagnostic only. Avoid scoring on independent fresh until freeze.
    with torch.no_grad():_,pilot=evaluate(np.arange(1840,2080))
    nparams=sum(p.numel() for p in net.parameters())
    row={'mode':mode,'seed':seed,'n_parameters':nparams,'best_epoch':bestep,
         'best_val_nll':best,'old_C64_test_nll_exploratory':float(pilot.mean()),
         'seconds':time.time()-start,'checkpoint':str(ck)}
    (output/f'{mode}_seed{seed}_log.json').write_text(json.dumps({'result':row,'epochs':records},indent=2))
    print(json.dumps(row),flush=True)
    return row

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--modes',default='Fisher,RankBalanced,HMComplementary,HM_Route,HM_ShuffledRoute,Fisher_Route');p.add_argument('--seed',type=int,default=641133);p.add_argument('--epochs',type=int,default=32)
    opt=p.parse_args(); rows=[]
    for mode in opt.modes.split(','):
        rows.append(train(mode,opt.seed,opt.epochs))
    d=pd.DataFrame(rows);d.to_csv(ROOT/f'validation_seed{opt.seed}.csv',index=False)
    print('VALIDATION_RANK',d.sort_values('best_val_nll')[['mode','best_val_nll','best_epoch']].to_json(orient='records'),flush=True)
