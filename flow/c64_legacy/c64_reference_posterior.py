#!/usr/bin/env python3
"""C64 foundation: 512D original C32 physics-based exact-likelihood MH.
Analytic tangent-Gaussian proposals with *nonlinear* Metropolis correction.
This is an uncertainty baseline, NOT a learned generative AI method.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'c32_original'))
import biot_mechanism_audit as B
import c32_linked_storage_audit as C


def sample_gaussian(mu,A,cho,rng):
    v=rng.standard_normal(512)
    eps=rng.standard_normal(A.shape[0])
    return mu+v-A.T@cho_solve(cho,A@v+eps,check_finite=False)

def log_q_unscaled(z,mu,A):
    d=z-mu
    return -.5*(d@d+(A@d)@(A@d))

def exact_logpost(z,y,rows,sigma,base,load):
    m=base+B.TM@z
    fy=C.forward(m[:256],m[256:],.7,load=load)[0]
    r=(fy[rows]-y[rows])/sigma[rows]
    return -.5*(z@z+r@r),fy

def solve_map(y,rows,sigma,base,load):
    def fun(z):
        m=base+B.TM@z
        fy,J,_=C.forward(m[:256],m[256:],.7,load=load,jac=True)
        r=(fy[rows]-y[rows])/sigma[rows]
        A=(J[rows,:512]@B.TM)/sigma[rows,None]
        return .5*(z@z+r@r),z+A.T@r
    opt=minimize(fun,np.zeros(512),jac=True,method='L-BFGS-B',options={'maxiter':120,'gtol':1e-7,'ftol':1e-13,'maxls':30})
    f,g=fun(opt.x)
    return opt.x,{'objective':float(f),'grad_norm':float(np.linalg.norm(g)),'success':bool(opt.success),'niter':int(opt.nit),'nfev':int(opt.nfev)}

def effective_sample_size(a):
    x=np.asarray(a,dtype=float)
    if x.ndim==1:x=x[:,None]
    out=[]
    for j in range(x.shape[1]):
        v=x[:,j];v=v-np.mean(v);n=len(v)
        var=np.dot(v,v)
        if var<1e-20:out.append(0.);continue
        # Geyer initial positive paired autocorrelation sequence
        maxlag=min(n//3,500); rho=[]
        for k in range(1,maxlag+1):rho.append(float(np.dot(v[:-k],v[k:])/var))
        tau=1.
        for k in range(0,len(rho)-1,2):
            pair=rho[k]+rho[k+1]
            if pair<=0:break
            tau+=2*pair
        out.append(float(n/tau))
    return out

def run(args):
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    base=np.r_[B.basemap(args.scenario)] if False else np.concatenate(B.basemap(args.scenario))
    load='reversal' if args.scenario=='channel' else 'pulse'
    rng=np.random.default_rng(args.seed)
    ztrue=rng.standard_normal(512)
    mtrue=base+B.TM@ztrue
    yt=C.forward(mtrue[:256],mtrue[256:],.7,load=load)[0]
    yr=C.forward(*B.basemap('homogeneous'),.7,load='pulse')[0]
    sigma=np.zeros(90)
    for t in range(3):
        sigma[t*30:t*30+14]=max(.005,float(np.std(yr[t*30:t*30+14]))*.075)*args.noise_multiplier
        sigma[t*30+14:t*30+30]=max(.001,float(np.std(yr[t*30+14:t*30+30]))*.075)*args.noise_multiplier
    y=yt+sigma*rng.standard_normal(90)
    meta={'scenario':args.scenario,'noise_multiplier':args.noise_multiplier,'seed':args.seed,'n_material_unknowns':512,'n_observations':90,'n_steps':args.steps,'burn':args.burn,'source':'original C32 M1-linked Biot PDE'}
    allresults={}
    for mode,rows in [('H',B.IH),('M',B.IM)]:
        if args.mode != 'both' and mode != args.mode: continue
        t0=time.time()
        mu,optim=solve_map(y,rows,sigma,base,load)
        mm=base+B.TM@mu
        fm,J,_=C.forward(mm[:256],mm[256:],.7,load=load,jac=True)
        A=J[rows,:512]@B.TM/sigma[rows,None]
        cov_ch=cho_factor(np.eye(len(rows))+A@A.T,lower=True,check_finite=False)
        # Exact Gauss-Newton tangent-Gaussian proposal centered on converged MAP
        # Gaussian prior N(0,I); local curvature I+A.T@A
        def glog(z):return log_q_unscaled(z,mu,A)
        # Independent Metropolized Gaussian proposals targeting original nonlinear posterior.
        chains=[];accs=[];phys_count=0
        for c in range(args.chains):
            crng=np.random.default_rng(args.seed+1000*(c+1)+ (1 if mode=='H' else 2))
            z=mu.copy(); lp,_=exact_logpost(z,y,rows,sigma,base,load); lq=glog(z)
            vals=[]; accepted=0
            for i in range(args.steps):
                zp=sample_gaussian(mu,A,cov_ch,crng)
                lpp,_=exact_logpost(zp,y,rows,sigma,base,load)
                lqp=glog(zp)
                if np.log(crng.random())<min(0,lpp-lp+lq-lqp):
                    z=zp;lp=lpp;lq=lqp;accepted+=1
                if i>=args.burn and (i-args.burn)%args.thin==0:
                    vals.append(z.copy())
            chains.append(np.asarray(vals));accs.append(accepted/args.steps)
        zpost=np.concatenate(chains)
        mp=base+zpost@B.TM.T
        # Compare Gaussian vs Metropolis-corrected nonlinear posterior, same observation.
        gauss_rng=np.random.default_rng(args.seed+765432+(0 if mode=='H' else 8))
        zg=np.stack([sample_gaussian(mu,A,cov_ch,gauss_rng) for _ in range(len(zpost))])
        mg=base+zg@B.TM.T
        diagC=1-np.sum(A*cho_solve(cov_ch,A,check_finite=False),axis=0)
        # physical posterior predictive residual using a limited, *real* original PDE forward batch
        pred_n=min(64,len(zpost));picks=np.linspace(0,len(zpost)-1,pred_n,dtype=int)
        post_yp=np.stack([C.forward(m[:256],m[256:],.7,load=load)[0][rows] for m in mp[picks]])
        gauss_yp=np.stack([C.forward(m[:256],m[256:],.7,load=load)[0][rows] for m in mg[picks]])
        truth_rmse=lambda p:[float(np.sqrt(np.mean((p[:256]-mtrue[:256])**2))),float(np.sqrt(np.mean((p[256:]-mtrue[256:])**2)))]
        result={'mode':mode,'rows':len(rows),'MAP':optim,'acceptance':accs,'n_retained':len(zpost),'posterior_ESS_z0_z1_z2':effective_sample_size(zpost[:,:3]),
            'logk_logE_rmse_MAP':truth_rmse(mm), 'logk_logE_rmse_nonlin_mean':truth_rmse(mp.mean(axis=0)),
            'logk_logE_rmse_tangent_mean':truth_rmse(mg.mean(axis=0)),
            'mean_gap_logk_logE':[float(np.sqrt(np.mean((mp.mean(axis=0)[sl]-mg.mean(axis=0)[sl])**2))) for sl in [slice(None,256),slice(256,None)]],
            'predictive_obs_whitened_rmse_nonlinear':float(np.sqrt(np.mean(((post_yp-y[rows])/sigma[rows])**2))),
            'predictive_obs_whitened_rmse_tangent':float(np.sqrt(np.mean(((gauss_yp-y[rows])/sigma[rows])**2))),
            'posterior_std_logk_logE':[float(np.mean(mp[:,sl].std(axis=0,ddof=1))) for sl in [slice(None,256),slice(256,None)]],
            'tangent_std_logk_logE':[float(np.mean(mg[:,sl].std(axis=0,ddof=1))) for sl in [slice(None,256),slice(256,None)]],
            'local_information_trace':float(512-diagC.sum()),
            'runtime_seconds':float(time.time()-t0),
            'remark':'nonlinear MH produces samples from the exact discrete-posterior as stationary distribution; finite-chain ESS and burn are diagnostic, not a convergence guarantee'}
        np.savez_compressed(out/f'{mode}_posterior_samples.npz',z_posterior=zpost.astype(np.float32),z_tangent=zg.astype(np.float32),m_posterior=mp.astype(np.float32),m_tangent=mg.astype(np.float32),MAP=mm.astype(np.float32),true_m=mtrue.astype(np.float32),y=y.astype(np.float32),sigma=sigma.astype(np.float32),rows=rows)
        allresults[mode]=result
        print(json.dumps(result,ensure_ascii=False),flush=True)
    (out/'results.json').write_text(json.dumps({'config':meta,'results':allresults},indent=2,ensure_ascii=False))
    return allresults

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',default='both',choices=['H','M','both'])
    p.add_argument('--scenario',default='homogeneous',choices=['homogeneous','channel','barrier'])
    p.add_argument('--noise-multiplier',type=float,default=1.)
    p.add_argument('--steps',type=int,default=2400)
    p.add_argument('--burn',type=int,default=400)
    p.add_argument('--thin',type=int,default=4)
    p.add_argument('--chains',type=int,default=2)
    p.add_argument('--seed',type=int,default=641101)
    p.add_argument('--outdir',default=str(HERE/'results'))
    run(p.parse_args())
