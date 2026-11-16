#!/usr/bin/env python3
"""C32 minimal physical-consistency audit. Based on preserved V2.9 independent solver.
Provides reference-matched M1-frozen vs M1-linked, 512D field tangent,
alpha sensitivity, nonlinear compensation; reference no field-data claims.
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.linalg import cho_factor, cho_solve
import biot_mechanism_audit as B

NU=B.NU; NP=B.NP; NOBS=90; dt=B.DT
PHI=0.20; NU_POISSON=B.nu; ALPHA0=.7
# Set fluid term so that reference E=1, alpha=.7 corresponds to S=.8.
CF=3*(1-2*NU_POISSON)
FLUID_TERM=.8-CF*(ALPHA0-PHI)*(1-ALPHA0)
K_F=PHI/FLUID_TERM

def storage(me, alpha):
    E=np.exp(me)
    return FLUID_TERM+CF*(alpha-PHI)*(1-alpha)/E

def storage_dlogE(me,alpha):
    return -CF*(alpha-PHI)*(1-alpha)/np.exp(me)

def storage_dalpha(me,alpha):
    return CF*(1+PHI-2*alpha)/np.exp(me)

def get_storage(mode, me, alpha, frozen_S):
    if mode=='M0': return np.full(NP,B.S),np.zeros(NP),np.zeros(NP)
    if mode=='M1-frozen':
        if frozen_S is None: raise ValueError('frozen_S must be provided')
        return frozen_S,np.zeros(NP),np.zeros(NP)
    if mode=='M1-linked':return storage(me,alpha), storage_dlogE(me,alpha),storage_dalpha(me,alpha)
    raise ValueError(mode)

def forward(mk, me, alpha, mode='M1-linked', frozen_S=None, load='pulse', jac=False):
    """Monolithic transient Biot; Jraw columns: 256 logk, 256 logE, 1 alpha.
    State-discretization and source/sensor geometry inherited from V2.9.
    """
    S,dSdE,dSdalpha=get_storage(mode,me,alpha,frozen_S)
    A0,G,lam,edgek=B.build_matrix(mk,me,alpha)
    # Replace original constant S/DT with a spatial diagonal storage.
    A=(A0+sp.diags(np.r_[np.zeros(2*NU),(S[B.interior]-B.S)/dt],format='csc')).tocsc()
    fac=splu(A)
    previous=np.zeros(3*NU)
    prior_tan=np.zeros((3*NU,2*NP+1)) if jac else None
    y=[];jac_rows=[];states=[]
    for t in range(B.STEPS):
        rhs=np.zeros(3*NU)
        src=np.zeros(NU)
        strength=([1.,.6,.2] if load=='pulse' else [1.,-.8,.7])[t]
        src[B.full_to_local[5*B.N+5]]=strength
        src[B.full_to_local[11*B.N+11]]=-.5*strength
        rhs[2*NU:]=src+(S[B.interior]/dt)*previous[2*NU:]+(alpha/dt)*(B.IP@B.DIV@previous[:2*NU])
        v=fac.solve(rhs)
        u=v[:2*NU];p=v[2*NU:]
        y.append(np.r_[p[B.hindex],u[B.mindex],u[NU+B.mindex]])
        states.append(v)
        if jac:
            RE=(B.EX.T@sp.diags(2*G*(B.EX@u)) + B.EY.T@sp.diags(2*G*(B.EY@u)) +
                B.SH.T@sp.diags(4*G*(B.SH@u)) + B.DIV.T@sp.diags(lam*(B.DIV@u))).toarray()
            RK=(B.IncP.T@sp.diags(edgek*(B.IncP@p))@B.Mean).toarray()
            r=np.zeros((3*NU,2*NP+1))
            r[:2*NU,NP:2*NP]=-RE
            r[2*NU:,:NP]=-RK
            # Pointwise storage derivative: -(p_t-p_{t-1})*dS/dlogE/dt
            rr=np.arange(NU)
            r[2*NU+rr,NP+B.interior] += -(p-previous[2*NU:])*dSdE[B.interior]/dt
            r[2*NU:,-1] += -(p-previous[2*NU:])*dSdalpha[B.interior]/dt
            # Alpha derivative: top +C p, bottom -C^T (u_t-u_prev)/dt
            C=(B.DIV.T@B.Sel).tocsc()
            r[:2*NU,-1]+=(C@p)
            r[2*NU:,-1]-=(C.T@(u-previous[:2*NU]))/dt
            r[2*NU:,:]+=(S[B.interior,None]/dt)*prior_tan[2*NU:,:]+(alpha/dt)*(B.IP@B.DIV@prior_tan[:2*NU,:])
            sens=fac.solve(r)
            jac_rows.append(np.vstack([sens[2*NU+B.hindex,:],sens[B.mindex,:],sens[NU+B.mindex,:]]))
            prior_tan=sens
        previous=v
    return np.concatenate(y), (np.vstack(jac_rows) if jac else None),states

def errors(mk,me,alpha,mode,frozenS,Jraw,rng,eps=1e-4):
    e=[]
    for _ in range(3):
        d=rng.normal(size=2*NP);d/=np.linalg.norm(d)
        dmk,dme=d[:NP],d[NP:]
        yp=forward(mk+eps*dmk,me+eps*dme,alpha,mode,frozenS)[0]
        ym=forward(mk-eps*dmk,me-eps*dme,alpha,mode,frozenS)[0]
        fd=(yp-ym)/(2*eps);ad=Jraw[:,:2*NP]@d
        e.append(float(np.linalg.norm(fd-ad)/max(np.linalg.norm(fd),1e-15)))
    yp=forward(mk,me,alpha+eps,mode,frozenS)[0]
    ym=forward(mk,me,alpha-eps,mode,frozenS)[0]
    fd=(yp-ym)/(2*eps);ad=Jraw[:,-1]
    ea=float(np.linalg.norm(fd-ad)/max(np.linalg.norm(fd),1e-15))
    return e,ea

def compensation(mk,me,alpha,mode,frozenS,y,Jraw,noise,da=.025):
    # Correct whitened-prior compensation: physical field shift = TM * dw.
    A=(Jraw[:,:2*NP]@B.TM)/noise[:,None]
    b=Jraw[:,-1]/noise
    out={}
    for key,ids in [('H',B.IH),('M',B.IM),('HM',np.arange(NOBS))]:
        AA=A[ids,:];bb=b[ids]
        c=cho_factor(np.eye(len(ids))+AA@AA.T,lower=True,check_finite=False)
        dw=-AA.T@cho_solve(c,bb,check_finite=False)*da
        shift=B.TM@dw
        ya=forward(mk,me,alpha+da,mode,frozenS)[0]
        ycomp=forward(mk+shift[:NP],me+shift[NP:],alpha+da,mode,frozenS)[0]
        only=np.linalg.norm((ya[ids]-y[ids])/noise[ids])
        both=np.linalg.norm((ycomp[ids]-y[ids])/noise[ids])
        out[key+'_ratio']=float(both/only)
        out[key+'_material_step_norm']=float(np.linalg.norm(dw))
    return out

def audit(outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    t0=time.time();rng=np.random.default_rng(20263211);records=[]
    ref_h,_,_=B.forward(*B.basemap('homogeneous'),alpha=.7)
    noise=np.empty(90)
    for t in range(B.STEPS):
        noise[t*30:t*30+14]=max(.005,float(np.std(ref_h[t*30:t*30+14]))*.075)
        noise[t*30+14:t*30+30]=max(.001,float(np.std(ref_h[t*30+14:t*30+30]))*.075)
    for case in ['channel','barrier']:
        mk,me=B.basemap(case)
        freeze=storage(me,ALPHA0)
        results={}
        for mode in ['M0','M1-frozen','M1-linked']:
            y,j,states=forward(mk,me,ALPHA0,mode,freeze,jac=True)
            em,ea=errors(mk,me,ALPHA0,mode,freeze,j,rng)
            item={'case':case,'mode':mode,'storage_min':float(min(get_storage(mode,me,ALPHA0,freeze)[0])),
                'storage_max':float(max(get_storage(mode,me,ALPHA0,freeze)[0])),
                'tangent_error_max':max(em),'alpha_sens_error':ea,
                'alpha_sens_norm':float(np.linalg.norm(j[:,-1]/noise)),
                'jacobian_fro':float(np.linalg.norm(j[:,:512]@B.TM/noise[:,None]))}
            item.update(compensation(mk,me,ALPHA0,mode,freeze,y,j,noise))
            results[mode]=y
            records.append(item)
            print(json.dumps(item),flush=True)
            np.savez_compressed(outdir/f'{case}_{mode}.npz',response=y,Jacobian_w=j[:,:512]@B.TM,
                Jacobian_alpha=j[:,-1],storage=get_storage(mode,me,ALPHA0,freeze)[0],sigma=noise,
                baseline_logk=mk,baseline_logE=me)
        # exact reference matching M1-frozen and linked, including three time steps
        err=float(np.linalg.norm(results['M1-frozen']-results['M1-linked'])/
                  max(np.linalg.norm(results['M1-linked']),1e-15))
        print(case,'matched M1 forward relative error',err,flush=True)
        if err>1e-12:raise AssertionError('M1 reference matching failed')
    cols=list(records[0]);
    with (outdir/'c32_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(records)
    validity=bool(max(r['tangent_error_max'] for r in records)<2e-5 and
                  max(r['alpha_sens_error'] for r in records)<2e-5)
    m={'validation_pass':validity,'simulator_identity':'V2.9 independent numerical Biot solver modified for S(E,alpha)',
       'not_historical_C31_reproduction':True,'physical_status':'dimensionless synthetic and specific isotropic constitutive parameterization',
       'nu':B.nu,'phi':PHI,'Kf':K_F,'fluid_term_phi_over_Kf':FLUID_TERM,'alpha_ref':ALPHA0,
       'storage_reference_homogeneous_E1':float(storage(np.zeros(NP),ALPHA0)[0]),
       'n_material_unknowns':512,'n_observations':90,'time_steps':3,
       'cases':['channel','barrier'],'modes':['M0','M1-frozen','M1-linked'],
       'elapsed_seconds':time.time()-t0,'records':records}
    (outdir/'c32_manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False))
    if not validity:raise AssertionError('C32 derivative validation failed')
    return m
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--outdir',default='c32_results');args=a.parse_args();audit(args.outdir)
