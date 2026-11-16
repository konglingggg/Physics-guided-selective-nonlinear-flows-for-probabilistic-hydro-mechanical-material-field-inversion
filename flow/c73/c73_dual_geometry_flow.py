#!/usr/bin/env python3
"""C73 stage B: dual-aligned physics context + material Fisher basis.
All 512D material variables retained; exactly 302848 trainable parameters.
Context transformations are fixed, physically calculated from C32 full 90x512 Jacobian.
JointSVD: diagonalize joint linear Biot observation response;
HMBlockSVD: separately align 42 H and 48 M observations;
Whiten: also normalize tangent predictive measurement variance;
RandomCtx: matched parameter-free orthogonal alternative.
Train and select strictly on original 1600/240 train-validation; fresh test later.
"""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
from pathlib import Path
import sys,time,json,argparse
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import c73_route_flow as CF
G=CF.G
D=CF.D
A=CF.A
W=CF.V
z=CF.z;mu=CF.mu;ctx=CF.ctx
Ujoint,sjoint,_=np.linalg.svd(A,full_matrices=False)
Uh,sh,_=np.linalg.svd(CF.AH,full_matrices=False)
Um,sm,_=np.linalg.svd(CF.AM,full_matrices=False)
UHM=np.zeros((90,90));UHM[:42,:42]=Uh;UHM[42:,42:]=Um
sHM=np.r_[sh,sm]
# Strong controls that keep exactly the same 90 observation coordinates.
Zpred=np.eye(90)+A@A.T
sensor_pred_var=np.diag(Zpred)
rg=np.random.default_rng(731562)
QrH,_=np.linalg.qr(rg.standard_normal((42,42)))
QrM,_=np.linalg.qr(rg.standard_normal((48,48)))
Urndblock=np.zeros((90,90));Urndblock[:42,:42]=QrH;Urndblock[42:,42:]=QrM
rnd_pred_var=np.diag(Urndblock.T@Zpred@Urndblock)
rng=np.random.default_rng(731561);Qr,_=np.linalg.qr(rng.standard_normal((90,90)))
TRANSFORMS={
    'JointSVD':Ujoint,
    'HMBlockSVD':UHM,
    'RandomCtx':Qr,
    # input ctx = physical residual r/10; make normalized predictive coordinate var~1:
    'JointSVDWhiten':Ujoint*(10/np.sqrt(1+sjoint**2))[None,:],
    'HMBlockSVDWhiten':UHM*(10/np.sqrt(1+sHM**2))[None,:],
    'SensorDiagNorm':np.diag(10/np.sqrt(sensor_pred_var)),
    'RandomBlockWhiten':Urndblock*(10/np.sqrt(rnd_pred_var))[None,:],
}

def train(mode,seed=641133,epochs=32,batch=80):
    t0=time.time();torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    Z=torch.from_numpy((z@W).astype('float32'))
    M=torch.from_numpy((mu@W).astype('float32'))
    CT=torch.from_numpy((ctx@TRANSFORMS[mode]).astype('float32'))
    AR=torch.from_numpy((A@W).astype('float32'))
    net=G.CondFlow(90)
    opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=5e-5)
    def loss(idx,reg=False):
        x=Z[idx];m=M[idx];c=CT[idx]
        u,ld=net.inverse(x,c,m);d=u-m
        n=.5*d.square().sum(1)+.5*(d@AR.T).square().sum(1)-ld-.5*CF.logdet
        return n.mean()+(.3*(u-x).square().mean() if reg else 0),n
    best=1e12;bestep=0;epochslog=[]
    out=ROOT/'dual_checkpoints';out.mkdir(exist_ok=True)
    ck=out/f'{mode}_seed{seed}.pt'
    for ep in range(1,epochs+1):
        net.train();pp=rng.permutation(1600)
        for i in range(0,1600,batch):
            ids=pp[i:i+batch];opt.zero_grad();l,_=loss(ids,True)
            l.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.5);opt.step()
        net.eval()
        with torch.no_grad():v,_=loss(np.arange(1600,1840))
        val=float(v)
        epochslog.append({'epoch':ep,'val_nll':val})
        if val<best-1e-5:
            best=val;bestep=ep;torch.save(net.state_dict(),ck)
    net.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu'));net.eval()
    with torch.no_grad():_,pilot=loss(np.arange(1840,2080))
    row={'mode':mode,'seed':seed,'n_parameters':sum(p.numel() for p in net.parameters()),
        'best_epoch':bestep,'val_nll':best,'old_C64_pilot_nll':float(pilot.mean()),
        'seconds':time.time()-t0,'checkpoint':str(ck)}
    (out/f'{mode}_seed{seed}_log.json').write_text(json.dumps({'summary':row,'training':epochslog},indent=2))
    print('TRAIN',json.dumps(row),flush=True)
    return row
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--modes',default=','.join(TRANSFORMS));q.add_argument('--seed',type=int,default=641133)
    p=q.parse_args();rows=[]
    for name in p.modes.split(','):rows.append(train(name,p.seed))
    pd.DataFrame(rows).to_csv(ROOT/f'dual_validation_seed{p.seed}.csv',index=False)
    print('DUAL_VALIDATION',pd.DataFrame(rows).sort_values('val_nll')[['mode','val_nll']].to_json(orient='records'),flush=True)
