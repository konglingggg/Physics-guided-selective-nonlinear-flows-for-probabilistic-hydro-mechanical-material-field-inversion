#!/usr/bin/env python3
"""C74-P5 exact nonlinear joint-Biot likelihood pilot, four independent observations.
Frozen models. New Biot observations were generated without model selection.
Per observation: converged nonlinear MAP + local 90x512 tangent Gaussian,
then center the frozen C73/P4 or full-coordinate learned bijection at MAP.
Every method uses same local MAP covariance, independent latent draws, 96 exact PDE evaluations.
ESS means importance-weight concentration only, NOT converged MCMC posterior.
"""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,json,time
from pathlib import Path
import numpy as np,torch,pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import c74_p5_strong_baselines as T
DG=T.D
C=DG.CF.JF.C
B=DG.CF.JF.B
DAT=DG.D
torch.set_num_threads(1)


def get_map(y,start=None):
    sigma=DAT['sigma'].astype(float);base=DAT['base'].astype(float)
    calls={'jac':0,'forward':0}
    def fun(z):
        m=base+B.TM@z
        f,J,_=C.forward(m[:256],m[256:],.7,load='pulse',jac=True)
        calls['jac']+=1
        r=(f-y)/sigma
        A=J[:,:512]@B.TM/sigma[:,None]
        return .5*(z@z+r@r),z+A.T@r
    t0=time.time()
    x0=np.zeros(512) if start is None else start
    opt=minimize(fun,x0,jac=True,method='L-BFGS-B',options={'maxiter':100,'gtol':1e-7,'ftol':1e-12,'maxls':25})
    z=opt.x.copy()
    trace=[]
    for i in range(10):
        m=base+B.TM@z
        f,J,_=C.forward(m[:256],m[256:],.7,load='pulse',jac=True);calls['jac']+=1
        A=J[:,:512]@B.TM/sigma[:,None]
        r=(f-y)/sigma
        g=z+A.T@r; norm=float(np.linalg.norm(g));val=float(.5*(z@z+r@r))
        trace.append({'step':i,'objective':val,'gradient_norm':norm})
        if norm<2e-5:break
        chol=cho_factor(np.eye(90)+A@A.T,lower=True,check_finite=False)
        dz=-g+A.T@cho_solve(chol,A@g,check_finite=False)
        alpha=1.
        while alpha>2**-18:
            zp=z+alpha*dz;mp=base+B.TM@zp
            fp=C.forward(mp[:256],mp[256:],.7,load='pulse')[0]
            calls['forward']+=1
            rr=(fp-y)/sigma;vv=.5*(zp@zp+rr@rr)
            if vv<val-1e-9*alpha*norm**2:break
            alpha*=.5
        if alpha<=2**-18:break
        z=zp
    mm=base+B.TM@z
    f,J,_=C.forward(mm[:256],mm[256:],.7,load='pulse',jac=True);calls['jac']+=1
    A=J[:,:512]@B.TM/sigma[:,None]
    rr=(f-y)/sigma
    grad=float(np.linalg.norm(z+A.T@rr))
    meta={'objective':float(.5*(z@z+rr@rr)),'grad_norm':grad,'lbfgs_success':bool(opt.success),
          'lbfgs_iterations':int(opt.nit),'calls':calls,'seconds':time.time()-t0,'gn_trace':trace}
    return z,A,meta

@torch.no_grad()
def propose_and_logq(zmap,d,zcond,cond,mode,seed=641133,netcache=None):
    if mode=='MAPGaussian':
        delta=d.reshape(-1)
        z=zmap+delta
        return z,float(-.5*(delta@delta+(netcache['A']@delta)@(netcache['A']@delta)))
    W=DG.W if mode=='C73' else T.P4.W
    net=netcache['nets'][mode]
    c=torch.from_numpy(cond.astype('float32'))
    mu0=torch.from_numpy((zcond@W).astype('float32'))
    x_mu=net.forward(mu0,c,mu0)
    dw=torch.from_numpy((d@W).astype('float32'))
    x=net.forward(mu0+dw,c,mu0)
    mapw=torch.from_numpy((zmap@W).astype('float32'))
    samplew=mapw+(x-x_mu)
    candidate=(samplew.numpy()@W.T).astype(float)
    virtual=samplew-mapw+x_mu
    u,ld=net.inverse(virtual,c,mu0)
    delta=(u-mu0).numpy()[0].astype(float)
    A_W=netcache['A']@W
    logq=-.5*(delta@delta+(A_W@delta)@(A_W@delta))+float(ld[0])
    return candidate[0],logq

if __name__=='__main__':
    with np.load(HERE/'cohort_fresh1024_nominal.npz') as obj:
        zs=obj['z'];ys=obj['y']
    n=96;cases=[int(v) for v in os.environ.get('C74_P5_CASES','3,17,71,143').split(',')];methods=['MAPGaussian','C73','FisherSpline','FullSplineMatched']
    original_prior=DAT['base'];sigma=DAT['sigma'];rng=np.random.default_rng(744018)
    rows=[];weights=[];maps=[];cached={}
    for case in cases:
        y=ys[case].astype(float)
        zmap,A,meta=get_map(y)
        maps.append({'case':case,**meta})
        print('MAP',case,'grad',meta['grad_norm'],'iters',meta['lbfgs_iterations'],'cost',meta['calls'],flush=True)
        _,ctx=DG.CF.feat(y.reshape(1,-1))
        mu,_=DG.CF.feat(y.reshape(1,-1))
        ctx=(ctx@DG.TRANSFORMS['HMBlockSVDWhiten']).astype(np.float32)
        chol=cho_factor(np.eye(90)+A@A.T,lower=True,check_finite=False)
        v=rng.standard_normal((n,512));eps=rng.standard_normal((n,90))
        ds=v-(A.T@cho_solve(chol,(A@v.T+eps.T),check_finite=False)).T
        netcache={'A':A,'nets':{}}
        for mode in methods:
            if mode!='MAPGaussian':netcache['nets'][mode]=T.load_net(mode,641133)
            lws=[];zdraw=[]
            t0=time.time()
            for i in range(n):
                z,logq=propose_and_logq(zmap,ds[i:i+1],mu,ctx,mode,netcache=netcache)
                m=original_prior+B.TM@z
                fy=C.forward(m[:256],m[256:],.7,load='pulse')[0]
                resid=(fy-y)/sigma
                logpi=-.5*(z@z+resid@resid)
                lw=float(logpi-logq)
                lws.append(lw);zdraw.append(z)
                weights.append({'case':case,'method':mode,'sample':i,'logpi':float(logpi),'logq_unnormalized':logq,'logweight':lw})
            lw=np.array(lws)
            w=np.exp(lw-lw.max());w=w/w.sum()
            ess=float(1/np.sum(w*w))
            zmean=np.sum(np.array(zdraw)*w[:,None],axis=0)
            mm=original_prior+B.TM@zmean
            true=original_prior+B.TM@zs[case].astype(float)
            rk=float(np.sqrt(np.mean((mm[:256]-true[:256])**2)))
            re=float(np.sqrt(np.mean((mm[256:]-true[256:])**2)))
            rows.append({'case':case,'method':mode,'N':n,'ESS':ess,'ESS_fraction':ess/n,'max_w':float(w.max()),
                         'logk_weighted_RMSE':rk,'logE_weighted_RMSE':re,'n_exact_PDE':n,'seconds':time.time()-t0})
            print('PILOT',case,mode,'ess',round(ess,3),'maxw',round(float(w.max()),3),'E rmse',round(re,4),flush=True)
    pd.DataFrame(rows).to_csv(HERE/f'exact_posterior_cases_{"_".join(map(str,cases))}.csv',index=False)
    pd.DataFrame(weights).to_csv(HERE/f'exact_posterior_weights_{"_".join(map(str,cases))}.csv',index=False)
    (HERE/f'exact_posterior_map_{"_".join(map(str,cases))}.json').write_text(json.dumps(maps,indent=2))
    print('SUMMARY',pd.DataFrame(rows).groupby('method')[['ESS','logE_weighted_RMSE']].mean().to_string(),flush=True)
