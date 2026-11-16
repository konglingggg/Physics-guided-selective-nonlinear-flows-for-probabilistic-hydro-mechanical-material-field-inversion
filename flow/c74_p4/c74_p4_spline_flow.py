#!/usr/bin/env python3
"""C74-P4: full 512D physics-allocated vs random-allocated nonlinear spline couplings.

All methods share C73 Biot physics anchor, H/M-block-SVD observation whitening,
and a rank-balanced (fixed, orthogonal) Fisher material coordinate ordering.
The new P4 mechanism is WHERE 24 non-affine monotonic RQ splines (per active
coupling half) are allocated. Inactive dimensions retain affine coupling.
Physical alternatives: dominant Fisher eigenvalues, or 12 H-dominant + 12
M-dominant information directions per half. Matched random allocation control.
C73's original HMBlockSVDWhiten is held fixed and never retrained here.

NLL training on C64 original C32 Biot 1600 train / 240 validation; old C64 test
and all 2048-problem frozen C73 test sets are NOT used for model selection.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
P=Path(__file__).resolve().parent
C73=Path(os.environ.get('AI4THMC_C73_ROOT',str(P.parent/'c73')))
sys.path.insert(0,str(C73))
import c73_route_flow as CF
import c73_dual_geometry_flow as DG

K=5; B=4.0; N_SPLINE=24; HIDDEN=62
MIN_W=1e-3; MIN_H=1e-3; MIN_D=1e-3
W=CF.V[:,CF.perm_rank]
DATA_Z=torch.from_numpy((DG.z@W).astype('float32'))
DATA_MU=torch.from_numpy((DG.mu@W).astype('float32'))
DATA_CTX=torch.from_numpy((DG.ctx@DG.TRANSFORMS['HMBlockSVDWhiten']).astype('float32'))
AR=torch.from_numpy((DG.A@W).astype('float32'))


def physical_allocations(mode,seed=11011):
    ret=[]
    for half in range(2):
        global_ids=CF.perm_rank[half*256:(half+1)*256]
        if mode=='FisherSpline':
            local=np.argsort(-CF.ev[global_ids])[:N_SPLINE]
        elif mode=='HMInfoSpline':
            h=CF.h[global_ids];m=CF.m[global_ids];e=CF.ev[global_ids]
            ih=np.flatnonzero((h>m)&(e>1e-5));im=np.flatnonzero((h<=m)&(e>1e-5))
            assert len(ih)>=12 and len(im)>=12,(half,len(ih),len(im))
            local=np.r_[ih[np.argsort(-h[ih])[:12]],im[np.argsort(-m[im])[:12]]]
        elif mode=='RandomSpline':
            local=np.random.default_rng(seed+half).choice(256,N_SPLINE,replace=False)
        elif mode=='WeakSpline':
            local=np.argsort(CF.ev[global_ids])[:N_SPLINE]
        else: raise ValueError(mode)
        ret.append(np.sort(local).astype(np.int64))
    return ret


def rq_spline(inputs,unnorm_w,unnorm_h,unnorm_d,inverse=False,bound=B):
    """Monotone rational-quadratic spline, identity tails, analytic inverse/logdet.

    inputs shape (batch, dims), unnorm widths/heights (batch,dims,K),
    derivative logits (batch,dims,K-1). Returns outputs, sum log|det d(out)/d(in)|.
    Formula as in Neural Spline Flows, Durkan et al. (2019).
    """
    dtype=inputs.dtype
    widths=MIN_W+(1-K*MIN_W)*torch.softmax(unnorm_w,dim=-1)
    heights=MIN_H+(1-K*MIN_H)*torch.softmax(unnorm_h,dim=-1)
    # bin coordinates, each endpoint is +-bound
    cums_w=torch.cat((torch.zeros_like(widths[...,:1]),torch.cumsum(widths,dim=-1)),dim=-1)
    cums_h=torch.cat((torch.zeros_like(heights[...,:1]),torch.cumsum(heights,dim=-1)),dim=-1)
    cums_w=2*bound*cums_w-bound
    cums_h=2*bound*cums_h-bound
    w=widths*(2*bound);h=heights*(2*bound)
    derivatives=torch.cat((torch.ones_like(unnorm_d[...,:1]),MIN_D+torch.nn.functional.softplus(unnorm_d),torch.ones_like(unnorm_d[...,:1])),dim=-1)
    inside=(inputs>=-bound)&(inputs<=bound)
    inp=inputs.clamp(-bound+1e-6,bound-1e-6)
    bin_cums=cums_h if inverse else cums_w
    bin_idx=torch.searchsorted(bin_cums[...,1:-1].contiguous(),inp.contiguous().unsqueeze(-1)).squeeze(-1).clamp(0,K-1)
    def sel(x):return torch.gather(x,-1,bin_idx.unsqueeze(-1)).squeeze(-1)
    xk=sel(cums_w);yk=sel(cums_h)
    wk=sel(w);hk=sel(h)
    dk=sel(derivatives[...,:-1]); dk1=sel(derivatives[...,1:])
    delta=hk/wk
    if inverse:
        ydelta=inp-yk
        A=ydelta*(dk+dk1-2*delta)+hk*(delta-dk)
        BB=hk*dk-ydelta*(dk+dk1-2*delta)
        C=-delta*ydelta
        disc=(BB*BB-4*A*C).clamp_min(1e-18)
        theta=2*C/(-BB-torch.sqrt(disc))
        theta=theta.clamp(0.,1.)
    else:
        theta=((inp-xk)/wk).clamp(0.,1.)
    th1=1-theta
    den=delta+(dk+dk1-2*delta)*theta*th1
    deriv_num=delta.square()*(dk1*theta.square()+2*delta*theta*th1+dk*th1.square())
    logabsdet=torch.log(deriv_num.clamp_min(1e-18))-2*torch.log(den.clamp_min(1e-18))
    if inverse:
        output=xk+theta*wk; logabsdet=-logabsdet
    else:
        output=yk+hk*(delta*theta.square()+dk*theta*th1)/den
    return torch.where(inside,output,inputs),torch.where(inside,logabsdet,torch.zeros_like(logabsdet)).sum(-1)


class PartialSplineCoupling(nn.Module):
    def __init__(self,idx,swap=False,hidden=HIDDEN):
        super().__init__();self.swap=bool(swap)
        idx=np.asarray(idx,dtype=np.int64)
        rest=np.setdiff1d(np.arange(256,dtype=np.int64),idx)
        assert len(idx)==N_SPLINE
        self.register_buffer('spline_idx',torch.from_numpy(idx))
        self.register_buffer('affine_idx',torch.from_numpy(rest))
        noutput=len(rest)*2+len(idx)*(3*K-1)
        self.net=nn.Sequential(nn.Linear(256+90,hidden),nn.SiLU(),
                               nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,noutput))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Identical spline at initialization (equal bins; unit interior/end derivatives).
        deriv_bias=np.log(np.expm1(1-MIN_D))
        with torch.no_grad():
            bias=self.net[-1].bias
            base=len(rest)*2
            for i in range(len(idx)):
                a=base+i*(3*K-1)+2*K
                bias[a:a+K-1]=float(deriv_bias)
    def split(self,x):return (x[:,256:],x[:,:256]) if self.swap else (x[:,:256],x[:,256:])
    def merge(self,a,b):return torch.cat((b,a),dim=1) if self.swap else torch.cat((a,b),dim=1)
    def params(self,a,c):
        raw=self.net(torch.cat((a,c),dim=1))
        na=len(self.affine_idx);ns=len(self.spline_idx)
        rs,rt=raw[:,:na],raw[:,na:2*na]
        pq=raw[:,2*na:].reshape(len(a),ns,3*K-1)
        return .18*torch.tanh(rs),.20*torch.tanh(rt),pq[:,:,:K],pq[:,:,K:2*K],pq[:,:,2*K:]
    def _execute(self,x,c,inverse=False):
        a,b=self.split(x);s,t,pw,ph,pd=self.params(a,c)
        af=b.index_select(1,self.affine_idx)
        sp=b.index_select(1,self.spline_idx)
        if inverse:
            af2=(af-t)*torch.exp(-s)
        else: af2=af*torch.exp(s)+t
        sp2,lds=rq_spline(sp,pw,ph,pd,inverse=inverse)
        bout=b.index_copy(1,self.affine_idx,af2).index_copy(1,self.spline_idx,sp2)
        lda=(-s if inverse else s).sum(1)
        return self.merge(a,bout),lda+lds
    def forward(self,x,c):return self._execute(x,c,False)
    def inverse(self,x,c):return self._execute(x,c,True)

class PartialSplineFlow(nn.Module):
    def __init__(self,mode,seed=11011):
        super().__init__()
        inds=physical_allocations(mode,seed)
        self.layers=nn.ModuleList([PartialSplineCoupling(inds[1 if i%2==0 else 0],swap=bool(i%2)) for i in range(4)])
    def inverse(self,z,c,mu):
        x=z-mu;ld=torch.zeros(len(z),device=z.device)
        for layer in reversed(self.layers):
            x,v=layer.inverse(x,c);ld=ld+v
        return x+mu,ld
    def forward_with_ld(self,u,c,mu):
        x=u-mu;ld=torch.zeros(len(u),device=u.device)
        for layer in self.layers:
            x,v=layer.forward(x,c);ld=ld+v
        return x+mu,ld
    def forward(self,u,c,mu):return self.forward_with_ld(u,c,mu)[0]


def build_net(mode,seed):
    if mode=='BalancedAffine':return DG.G.CondFlow(90)
    return PartialSplineFlow(mode,seed=11011) # frozen allocation across training seeds


def nll_for(net,ids,with_reg=False):
    zz=DATA_Z[ids];mm=DATA_MU[ids];cc=DATA_CTX[ids]
    u,ld=net.inverse(zz,cc,mm)
    d=u-mm
    nll=.5*d.square().sum(1)+.5*(d@AR.T).square().sum(1)-ld-.5*DG.CF.logdet
    reg=.3*(u-zz).square().mean() if with_reg else nll.new_zeros(())
    return nll.mean()+reg,nll


def train(mode,seed,epochs=32,batch=80):
    start=time.time();torch.set_num_threads(1);torch.manual_seed(seed)
    rng=np.random.default_rng(seed);net=build_net(mode,seed)
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=5e-5)
    out=P/'checkpoints';out.mkdir(exist_ok=True)
    ckpt=out/f'{mode}_seed{seed}.pt';rows=[]
    best=float('inf');best_ep=0
    for ep in range(1,epochs+1):
        net.train();perm=rng.permutation(1600)
        for startidx in range(0,1600,batch):
            idx=perm[startidx:startidx+batch]
            optimizer.zero_grad()
            loss,_=nll_for(net,idx,True)
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite training loss {mode} {seed} {ep}')
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.5);optimizer.step()
        net.eval()
        with torch.no_grad():vl=float(nll_for(net,np.arange(1600,1840))[0])
        rows.append({'epoch':ep,'validation_nll':vl})
        if vl<best-1e-5:
            best=vl;best_ep=ep;torch.save(net.state_dict(),ckpt)
        if ep%8==0:print('EPOCH',mode,seed,ep,round(vl,6),'BEST',best_ep,round(best,6),flush=True)
    net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True));net.eval()
    with torch.no_grad():
        x=DATA_Z[1600:1608];c=DATA_CTX[1600:1608];mu=DATA_MU[1600:1608]
        u,ld=net.inverse(x,c,mu)
        if isinstance(net,PartialSplineFlow):
            x2,ld2=net.forward_with_ld(u,c,mu)
            inv_error=float((x2-x).abs().max()); ld_error=float((ld+ld2).abs().max())
        else:
            inv_error=float((net.forward(u,c,mu)-x).abs().max());ld_error=np.nan
    row={'mode':mode,'seed':seed,'n_parameters':sum(p.numel() for p in net.parameters()),
         'epochs':epochs,'best_epoch':best_ep,'val_nll':best,'train_seconds':round(time.time()-start,2),
         'inverse_error':inv_error,'ld_consistency_error':ld_error,'checkpoint':str(ckpt)}
    (out/f'{mode}_seed{seed}_history.json').write_text(json.dumps({'summary':row,'training':rows},indent=2))
    print('MODEL',json.dumps(row),flush=True)
    return row


def audit(seed=641133):
    torch.set_num_threads(1);torch.manual_seed(seed)
    net=PartialSplineFlow('HMInfoSpline')
    npar=sum(p.numel() for p in net.parameters())
    a=torch.randn(7,512)*1.4;c=torch.randn(7,90)*.2;mu=torch.randn(7,512)*.08
    z,ld_f=net.forward_with_ld(a,c,mu);a2,ld_i=net.inverse(z,c,mu)
    assert float((a-a2).abs().max())<1e-5
    assert float((ld_f+ld_i).abs().max())<2e-5
    # Controlled nonidentity parameters to stress-check RQS and analytic inverse.
    with torch.no_grad():
        for layer in net.layers:layer.net[-1].weight.normal_(0,.0005)
    z,ld_f=net.forward_with_ld(a,c,mu);a2,ld_i=net.inverse(z,c,mu)
    err=float((a-a2).abs().max());lderr=float((ld_f+ld_i).abs().max())
    assert err<2e-4,(err,lderr)
    assert lderr<2e-4,(err,lderr)
    # Analytic logdet check for a single transformed coordinate
    layer=net.layers[0]
    inp=torch.randn(1,512)*.7;cond=torch.randn(1,90)*.2
    x0,v0=layer.forward(inp,cond)
    idx=256+int(layer.spline_idx[0])
    eps=1e-3
    pp=inp.clone();pp[0,idx]+=eps
    pm=inp.clone();pm[0,idx]-=eps
    fp,_=layer.forward(pp,cond);fm,_=layer.forward(pm,cond)
    numerical_log=float(torch.log(((fp[0,idx]-fm[0,idx])/(2*eps)).abs()))
    # compare coordinatewise analytic logdet; isolate others by independently computing spline
    aa,bb=layer.split(inp);ss,tt,pw,ph,pd=layer.params(aa,cond)
    _,ld_coord=rq_spline(bb.index_select(1,layer.spline_idx)[:,:1],pw[:,:1],ph[:,:1],pd[:,:1])
    logerr=abs(numerical_log-float(ld_coord))
    assert logerr<.02,logerr
    info={'model_parameters':npar,'c73_parameters':302848,'parameter_difference_pct':100*(npar/302848-1),
          'stressed_inverse_error':err,'stressed_logdet_forward_inverse_error':lderr,
          'coordinate_logderivative_abs_error':logerr,'splines_per_half':N_SPLINE,'bins':K,
          'fisher_allocations':[a.tolist() for a in physical_allocations('FisherSpline')],
          'hm_allocations':[a.tolist() for a in physical_allocations('HMInfoSpline')]}
    (P/'architecture_audit.json').write_text(json.dumps(info,indent=2))
    print('AUDIT',json.dumps({k:v for k,v in info.items() if 'allocations' not in k}),flush=True)
    return info

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=641133)
    parser.add_argument('--modes',default='BalancedAffine,FisherSpline,HMInfoSpline,RandomSpline')
    parser.add_argument('--epochs',type=int,default=32);parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args();audit()
    if not args.audit_only:
        res=[train(m.strip(),args.seed,args.epochs) for m in args.modes.split(',') if m.strip()]
        pd.DataFrame(res).to_csv(P/f'validation_seed{args.seed}.csv',index=False)
