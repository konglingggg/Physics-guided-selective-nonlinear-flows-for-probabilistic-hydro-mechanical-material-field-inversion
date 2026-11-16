#!/usr/bin/env python3
"""C64: train 512-dimensional, H-only/M-only physics-anchored conditional RealNVP.
Trained by amortized posterior density estimation on (z,y) pairs from ORIGINAL C32.
Reports heldout logscore relative to full rank exact tangent Gaussian baseline.
The nonlinear Metropolis sampler is an independent physics posterior reference.
No H/M neural interaction.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'c32_original'))
import biot_mechanism_audit as B
import c32_linked_storage_audit as C

class Coupling(nn.Module):
    def __init__(self, nobs, hidden=80, swap=False):
        super().__init__();self.swap=swap
        self.net=nn.Sequential(nn.Linear(256+nobs,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,512))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def split(self,x):return (x[:,256:],x[:,:256]) if self.swap else (x[:,:256],x[:,256:])
    def merge(self,a,b):return torch.cat([b,a],dim=1) if self.swap else torch.cat([a,b],dim=1)
    def params(self,a,c):
        s,t=self.net(torch.cat([a,c],1)).chunk(2,dim=1)
        return .18*torch.tanh(s),.20*torch.tanh(t)
    def forward(self,x,c):
        a,b=self.split(x);s,t=self.params(a,c)
        return self.merge(a,b*torch.exp(s)+t),s.sum(1)
    def inverse(self,x,c):
        a,b=self.split(x);s,t=self.params(a,c)
        return self.merge(a,(b-t)*torch.exp(-s)),-s.sum(1)

class CondFlow(nn.Module):
    def __init__(self,nobs):
        super().__init__();self.layers=nn.ModuleList([Coupling(nobs,swap=bool(i%2)) for i in range(4)])
    def inverse(self,z,ctx,mu):
        x=z-mu;ld=torch.zeros(len(z),device=z.device)
        for layer in reversed(self.layers):
            x,d=layer.inverse(x,ctx);ld+=d
        return x+mu,ld
    def forward(self,z0,ctx,mu):
        x=z0-mu
        for layer in self.layers:x,_=layer(x,ctx)
        return x+mu

def cache_data(out,train,val,test,seed,noise):
    fname=out/'joint_dataset.npz'
    if fname.exists():
        a=np.load(fname);assert len(a['z'])==train+val+test;return fname
    rng=np.random.default_rng(seed)
    mk,me=B.basemap('homogeneous');base=np.r_[mk,me]
    yref=C.forward(mk,me,.7,load='pulse')[0]
    sigma=np.zeros(90)
    for t in range(3):
        sigma[t*30:t*30+14]=max(.005,float(np.std(yref[t*30:t*30+14]))*.075)*noise
        sigma[t*30+14:t*30+30]=max(.001,float(np.std(yref[t*30+14:t*30+30]))*.075)*noise
    zs=[];ys=[];t0=time.time()
    for i in range(train+val+test):
        z=rng.standard_normal(512)
        m=base+B.TM@z
        yn=C.forward(m[:256],m[256:],.7,load='pulse')[0]
        y=yn+sigma*rng.standard_normal(90)
        zs.append(z.astype('float32'));ys.append(y.astype('float32'))
        if (i+1)%500==0: print('generated',i+1,'time',round(time.time()-t0,2),flush=True)
    np.savez_compressed(fname,z=np.stack(zs),y=np.stack(ys),sigma=sigma,base=base,base_y=yref,
                        seed=seed,train=train,val=val,test=test)
    return fname

def init_physics(data,mode):
    rows=B.IH if mode=='H' else B.IM
    base=data['base']; sigma=data['sigma'];basey=data['base_y']
    _y,J,_=C.forward(base[:256],base[256:],.7,load='pulse',jac=True)
    A=(J[rows,:512]@B.TM)/sigma[rows,None]
    Z=np.eye(len(rows))+A@A.T
    ch=cho_factor(Z,lower=True,check_finite=False)
    def features(y):
        r=(y[:,rows]-basey[rows])/sigma[rows]
        mu=(r@cho_solve(ch,A,check_finite=False)).astype('float32')
        context=(r/10.).astype('float32')
        return mu,context
    logdet=np.linalg.slogdet(Z)[1]
    return A,features,logdet,rows

def train_one(data,mode,out,seed,train_n,val_n,epochs,batch):
    torch.set_num_threads(1);torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    A,get_feat,logdet,rows=init_physics(data,mode)
    z=data['z'];y=data['y'];mu,ctx=get_feat(y)
    z=torch.from_numpy(z);mu=torch.from_numpy(mu);ctx=torch.from_numpy(ctx)
    A=torch.tensor(A,dtype=torch.float32)
    net=CondFlow(len(rows)); opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=5e-5)
    def loss(idx,train=True):
        zz=z[idx];mm=mu[idx];cc=ctx[idx]
        z0,logdet_inv=net.inverse(zz,cc,mm)
        d=z0-mm
        nll=(.5*d.square().sum(1)+.5*(d@A.T).square().sum(1)-logdet_inv).mean()
        # centered at baseline (identity init), mild regularization
        reg=((z0-zz).square().mean())*.3 if train else nll.new_zeros(())
        return nll+reg,nll
    best=1e9;best_epoch=0;log=[];t0=time.time()
    n_train=len(z)-val_n-int(data['test'])
    for epoch in range(1,epochs+1):
        net.train();perm=rng.permutation(train_n)
        for i in range(0,train_n,batch):
            ids=perm[i:i+batch]
            opt.zero_grad();l,_=loss(ids);l.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.5);opt.step()
        net.eval()
        with torch.no_grad():
            _,v=loss(np.arange(train_n,train_n+val_n),False)
        log.append({'epoch':epoch,'val_nll_without_constant':float(v)})
        if float(v)<best-1e-5:
            best=float(v);best_epoch=epoch;torch.save(net.state_dict(),out/f'{mode}_conditional_flow.pt')
        if epoch%10==0:print(mode,'epoch',epoch,'val nll',float(v),'best',best_epoch,flush=True)
    net.load_state_dict(torch.load(out/f'{mode}_conditional_flow.pt',weights_only=True))
    net.eval();test_ids=np.arange(train_n+val_n,len(z))
    with torch.no_grad():
        _,test_nll=loss(test_ids,False)
        delta=z[test_ids]-mu[test_ids]
        base_nll=(.5*delta.square().sum(1)+.5*(delta@A.T).square().sum(1)).mean()
        _,indiv=loss(test_ids,False)
        # Posterior draws at an independent observation, using *exact* tangent base sample
        ind=int(test_ids[0]);rn=np.random.default_rng(seed+100)
        ch=cho_factor(np.eye(len(rows))+A.numpy()@A.numpy().T,lower=True,check_finite=False)
        Aa=A.numpy();mm=mu[ind].numpy();samples=[]
        for i in range(192):
            u=rn.standard_normal(512);v=rn.standard_normal(len(rows))
            dz=u-Aa.T@cho_solve(ch,Aa@u+v,check_finite=False)
            samples.append(mm+dz)
        tangent_samples=np.stack(samples).astype('float32')
        td=torch.tensor(tangent_samples)
        flow_samples=net.forward(td,ctx[ind].repeat(len(td),1),mu[ind].repeat(len(td),1)).numpy()
    np.savez_compressed(out/f'{mode}_generated_samples.npz',z_flow=flow_samples,z_tangent=tangent_samples,truth_z=z[ind].numpy(),y=y[ind],sigma=data['sigma'])
    result={'mode':mode,'nobs':len(rows),'train_pairs':train_n,'val_pairs':val_n,'test_pairs':len(test_ids),
            'best_epoch':best_epoch,'test_baseline_tangent_nll_wo_const':float(base_nll),
            'test_conditional_flow_nll_wo_const':float(test_nll),'nll_improvement_flow_minus_tangent':float(test_nll-base_nll),
            'n_parameters':sum(p.numel() for p in net.parameters()),'train_seconds':time.time()-t0,
            'disclaimer':'NLL is joint-predictive conditional proper log score at heldout simulated (m,y) pairs. Lower is better; not nonlinear exact posterior KL.'}
    (out/f'{mode}_training_log.json').write_text(json.dumps({'result':result,'epochs':log},indent=2))
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',type=int,default=1600);p.add_argument('--val',type=int,default=240);p.add_argument('--test',type=int,default=240)
    p.add_argument('--noise-multiplier',type=float,default=1.)
    p.add_argument('--epochs',type=int,default=32);p.add_argument('--batch',type=int,default=80);p.add_argument('--seed',type=int,default=641120)
    p.add_argument('--outdir',default=str(HERE/'flow_results'));args=p.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    zfile=cache_data(out,args.train,args.val,args.test,args.seed,args.noise_multiplier)
    with np.load(zfile) as dat:data={x:dat[x] for x in dat.files}
    results={}
    for mode in ['H','M']:results[mode]=train_one(data,mode,out,args.seed+(1 if mode=='H' else 2),args.train,args.val,args.epochs,args.batch)
    (out/'flow_summary.json').write_text(json.dumps(results,indent=2))
