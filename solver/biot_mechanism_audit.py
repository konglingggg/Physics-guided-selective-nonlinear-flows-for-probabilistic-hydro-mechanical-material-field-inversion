#!/usr/bin/env python3
"""C32 Biot solver - the reference implementation used for all reported results.

Physical discretization: 2-D square, 16x16 nodal fields, homogeneous Dirichlet
p and u at the perimeter, monolithic backward Euler 3 steps, linear isotropic
elastic skeleton with fixed Poisson's ratio (nu = 0.25); nonlinear dependence
on log k and log E.

This module is the canonical solver for every dataset in the manuscript. It was
originally written as an independent re-implementation that replaced the earlier
V-series pilot solver, and subsequently became the C32 baseline imported by the
entire C64 -> C69 -> C73 -> C74 analysis chain.

Entry points
------------
build_matrix(mk, me, alpha)  assemble the 588 x 588 coupled block system
forward(mk, me, alpha)       forward solve; returns y and the 90 x 512 Jacobian
check_tangent(mk, me, alpha) centered-difference audit behind Table 3 / Fig. 2
"""
import os, json, csv, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.ndimage import gaussian_filter
from scipy.linalg import cho_factor,cho_solve,svdvals
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent / 'reproduced'
OUT.mkdir(parents=True,exist_ok=True)
N=16; NP=N*N
NU=(N-2)*(N-2)
nu=.25; S=.8; DT=1.; STEPS=3; ALOGK=.55; ALOGE=.45
xs=[];ys=[]; edges=[]
for yy in range(N):
 for xx in range(N):
  i=yy*N+xx
  if xx<N-1: edges.append((i,i+1,0))
  if yy<N-1: edges.append((i,i+N,1))
Epairs=np.array([(i,j) for i,j,ax in edges],int)
M=len(edges)
rr=np.repeat(np.arange(M),2)
cc=Epairs.flatten(); dd=np.tile([1.,-1.],M)
Inc=sp.coo_matrix((dd,(rr,cc)),shape=(M,NP)).tocsr()
Gx=sp.coo_matrix((np.ones(M//2),(np.arange(M//2),np.zeros(M//2)))) if False else None
# Embedded grid row forward gradients for elastic energy at each nodal location
rows=[];cols=[];vals=[]
for y in range(N):
 for x in range(N-1):
  row=y*N+x
  rows+= [row,row]; cols+=[row,row+1]; vals +=[-1.,1.]
Dx=sp.coo_matrix((vals,(rows,cols)),shape=(NP,NP)).tocsr()
rows=[];cols=[];vals=[]
for y in range(N-1):
 for x in range(N):
  row=y*N+x
  rows +=[row,row]; cols +=[row,row+N]; vals +=[-1.,1.]
Dy=sp.coo_matrix((vals,(rows,cols)),shape=(NP,NP)).tocsr()
interior=np.array([y*N+x for y in range(1,N-1) for x in range(1,N-1)],int)
Sel=sp.coo_matrix((np.ones(NU),(interior,np.arange(NU))),shape=(NP,NU)).tocsr()
X=(Dx@Sel).tocsr(); Y=(Dy@Sel).tocsr()
zero=sp.csr_matrix((NP,NU))
EX=sp.hstack([X,zero]).tocsr()
EY=sp.hstack([zero,Y]).tocsr()
SH=sp.hstack([Y*.5,X*.5]).tocsr()
DIV=(EX+EY).tocsr()
IP=Sel.T.tocsr()
IncP=Inc@Sel
# average edge log coefficient derivative mapping each logk to incident edge via 1/2
av=np.repeat(np.arange(M),2)
Mean=sp.coo_matrix((np.full(2*M,.5),(av,cc)),shape=(M,NP)).tocsr()
# prior map: smooth Gaussian random field with row-wise normalized covariance factor
T0=np.empty((NP,NP))
for j in range(NP):
 u=np.zeros((N,N)); u.flat[j]=1.
 T0[:,j]=gaussian_filter(u,1.0,mode='reflect').ravel()
T=T0/np.maximum(np.sqrt(np.sum(T0**2,axis=1,keepdims=True)),1.e-15)
TM=np.zeros((2*NP,2*NP)); TM[:NP,:NP]=ALOGK*T; TM[NP:,NP:]=ALOGE*T
# 14 p and 8+8 displacement point readings per step -> 90
sensors_h=np.array([2*N+2,2*N+8,2*N+13,5*N+3,5*N+6,5*N+11,7*N+8,
 9*N+3,9*N+6,9*N+12,11*N+5,12*N+2,12*N+10,13*N+13])
sensors_m=np.array([3*N+3,3*N+7,3*N+12,5*N+9,7*N+4,8*N+12,11*N+6,12*N+11])
# full-grid sensor to interior index
full_to_local={v:i for i,v in enumerate(interior)}
hindex=np.array([full_to_local[x] for x in sensors_h])
mindex=np.array([full_to_local[x] for x in sensors_m])


def build_matrix(mk,me,alpha):
 k=np.exp(mk); E=np.exp(me)
 G=E/(2*(1+nu)); lam=E*nu/((1+nu)*(1-2*nu))
 def quadratic(B,coeff):return (B.T@sp.diags(coeff)@B).tocsc()
 Kuu=(quadratic(EX,2*G)+quadratic(EY,2*G)+quadratic(SH,4*G)+quadratic(DIV,lam)).tocsc()
 edgek=np.sqrt(k[Epairs[:,0]]*k[Epairs[:,1]])
 Lap=(IncP.T@sp.diags(edgek)@IncP).tocsc()
 C=(DIV.T@Sel).tocsc()  # (2nu,np), coupling
 A=sp.bmat([[Kuu,-alpha*C],[alpha/DT*C.T,(S/DT)*sp.eye(NU,format='csc')+Lap]],format='csc')
 return A,G,lam,edgek


def forward(mk,me,alpha,load='pulse',with_jac=False):
 A,G,lam,edgek=build_matrix(mk,me,alpha)
 fac=splu(A)
 previous=np.zeros(2*NU+NU)
 prior_tan=np.zeros((3*NU,2*NP)) if with_jac else None
 y=[]; jac=[]; states=[]
 for t in range(STEPS):
  rhs=np.zeros(3*NU)
  # localized source-sink pair changes in time, injection vs production
  src=np.zeros(NU)
  inj=full_to_local[5*N+5]; sink=full_to_local[11*N+11]
  strength=([1.,.6,.2] if load=='pulse' else [1.,-.8,.7])[t]
  src[inj]=strength; src[sink]=-.5*strength
  rhs[2*NU:]=src+(S/DT)*previous[2*NU:]+(alpha/DT)*(IP@DIV@previous[:2*NU])
  v=fac.solve(rhs)
  disp=v[:2*NU]; pressure=v[2*NU:]
  observations=np.concatenate([pressure[hindex],disp[mindex],disp[NU+mindex]])
  y.append(observations)
  states.append(v)
  if with_jac:
   # raw-log-parameter derivatives of K_uu(E)u and L(k)p
   RE=(EX.T@sp.diags(2*G*(EX@disp)) + EY.T@sp.diags(2*G*(EY@disp)) +
       SH.T@sp.diags(4*G*(SH@disp)) + DIV.T@sp.diags(lam*(DIV@disp))).toarray()
   # edge flux d ke/d logk_j = edgek*0.5 for each endpoint
   RK=(IncP.T@sp.diags(edgek*(IncP@pressure))@Mean).toarray()
   r=np.zeros((3*NU,2*NP))
   r[:2*NU,NP:]=-RE
   r[2*NU:,:NP]=-RK
   r[2*NU:,:]+=((S/DT)*prior_tan[2*NU:,:]+(alpha/DT)*(IP@DIV@prior_tan[:2*NU,:]))
   sens=fac.solve(r)
   j=np.vstack([sens[2*NU+hindex,:],sens[mindex,:],sens[NU+mindex,:]])
   jac.append(j)
   prior_tan=sens
  previous=v
 return np.concatenate(y), (np.vstack(jac) if with_jac else None), states


def basemap(kind):
 yy,xx=np.mgrid[:N,:N]
 mk=np.full((N,N),np.log(.15)); me=np.zeros((N,N))
 if kind=='barrier':
  # low-perm barrier x=7-9 with a narrow high-perm window near y=11
  band=np.logical_and(xx>=7,xx<=9)
  mk[band]-=2.6
  mk[np.logical_and(band,np.abs(yy-11)<=1)]+=2.0
  me[band]+=.45
 elif kind=='channel':
  path=np.abs(yy-(.65*xx+2))<1.6
  mk[path]+=1.8
  me[path]-=.35
 return mk.ravel(),me.ravel()


def get_split_rows():
 h=[];m=[]
 for j in range(STEPS):
  h+=list(range(j*30,j*30+14))
  m+=list(range(j*30+14,j*30+30))
 return np.array(h),np.array(m)
IH,IM=get_split_rows()

def pdmetric(J,y,truth, rows,noise):
 JJ=J[rows,:]/noise[rows,None]
 residual=y[rows]/noise[rows]
 gram=np.eye(len(rows))+JJ@JJ.T
 ch=cho_factor(gram,lower=True,check_finite=False)
 tmp=cho_solve(ch,residual,check_finite=False)
 mu=JJ.T@tmp
 cross=(JJ[:,:NP].T@cho_solve(ch,JJ[:,NP:],check_finite=False))
 # C cross block = -cross
 CkE=-cross
 # diag posterior variance, exact for whitened priors
 info_k=np.sum(JJ[:,:NP]*cho_solve(ch,JJ[:,:NP],check_finite=False),axis=0)
 info_E=np.sum(JJ[:,NP:]*cho_solve(ch,JJ[:,NP:],check_finite=False),axis=0)
 sk=np.sqrt(np.maximum(1.-info_k,1.e-12)); sE=np.sqrt(np.maximum(1.-info_E,1.e-12))
 ewhite=mu-truth
 errphysical=TM@ewhite
 normcross=np.linalg.norm(CkE,'fro')
 crossspect=svdvals(CkE)[0]
 ret={'info_k':float(np.sum(info_k)),'info_E':float(np.sum(info_E)),
 'rmse_wk':float(np.linalg.norm(ewhite[:NP])/np.sqrt(NP)),
 'rmse_wE':float(np.linalg.norm(ewhite[NP:])/np.sqrt(NP)),
 'rmse_logk':float(np.linalg.norm(errphysical[:NP])/np.sqrt(NP)),
 'rmse_logE':float(np.linalg.norm(errphysical[NP:])/np.sqrt(NP)),
 'coverage90_k':float(np.mean(np.abs(ewhite[:NP])<=1.6448536*sk)),
 'coverage90_E':float(np.mean(np.abs(ewhite[NP:])<=1.6448536*sE)),
 'fro_cross':float(normcross),'largest_cross_singular':float(crossspect),
 'mu':mu,'cross':CkE,'sd':np.r_[sk,sE],'J':JJ}
 return ret


def weighted_cross(J,noises):
 jk=J[:,:NP]/noises[:,None]; je=J[:,NP:]/noises[:,None]
 # normalized cross Gramians and similarity of influential response left-subspaces
 u1,s1,_=np.linalg.svd(jk,full_matrices=False)
 u2,s2,_=np.linalg.svd(je,full_matrices=False)
 r=5
 overlap=svdvals(u1[:,:r].T@u2[:,:r]) if s1[0]>1e-14 and s2[0]>1e-14 else None
 a=jk.T@je
 den=(np.linalg.norm(jk.T@jk,'fro')*np.linalg.norm(je.T@je,'fro'))**.5
 sc=np.linalg.norm(a,'fro')/den if den>1e-14 else 0.
 return {'top5_response_subspace_cosines':overlap.tolist() if overlap is not None else None,
 'top5_response_subspace_overlap_mean':float(np.mean(overlap)) if overlap is not None else None,
 'cross_gram_relative':float(sc),
 'singular_k_top3':s1[:3].tolist(),'singular_E_top3':s2[:3].tolist()}


def check_tangent(mk,me,alpha):
 rng=np.random.default_rng(1047)
 y,Jfield,_=forward(mk,me,alpha,with_jac=True)
 J=Jfield@TM
 errs=[]
 for i in range(4):
  d=rng.normal(size=2*NP);d/=np.linalg.norm(d)
  step=1.e-4
  dphys=TM@d
  yplus,_,_=forward(mk+step*dphys[:NP],me+step*dphys[NP:],alpha)
  ymin,_,_=forward(mk-step*dphys[:NP],me-step*dphys[NP:],alpha)
  approx=(yplus-ymin)/(2*step)
  pred=J@d
  errs.append(float(np.linalg.norm(approx-pred)/np.linalg.norm(approx)))
 return J,errs


def run():
 start=time.monotonic(); rows=[]; detailed={}; checks=[]
 mk,me=basemap('homogeneous')
 print('Analytic tangent vs central differences',flush=True)
 _,errs=check_tangent(mk,me,.7)
 print('tangent errors:',errs,flush=True)
 if max(errs)>1.e-5: raise RuntimeError('Tangent failed finite-difference test')
 cases=[('homogeneous','pulse',.7),('homogeneous','reversal',.7),
        ('barrier','pulse',.7),('barrier','reversal',.7),
        ('channel','pulse',.7),('channel','reversal',.7),
        ('homogeneous','pulse',.4),('homogeneous','pulse',1.),('homogeneous','pulse',0.)]
 # fixed noise based on homogeneous nominal reference, never rescale case by case
 ref,_,_=forward(mk,me,.7)
 noise=np.empty(90)
 for t in range(STEPS):
  noise[t*30:t*30+14]=max(.005,float(np.std(ref[t*30:t*30+14]))*.075)
  noise[t*30+14:t*30+30]=max(.001,float(np.std(ref[t*30+14:t*30+30]))*.075)
 print('fixed observation sigma H, M',noise[0],noise[14],flush=True)
 rng=np.random.default_rng(260911)
 for idx,(kind,load,alpha) in enumerate(cases):
  mk0,me0=basemap(kind)
  y0,Jraw,_=forward(mk0,me0,alpha,load,with_jac=True)
  J=Jraw@TM
  wtruth=rng.standard_normal(2*NP)
  delta=TM@wtruth
  ytrue,_,_=forward(mk0+delta[:NP],me0+delta[NP:],alpha,load)
  noisy=ytrue+rng.normal(size=90)*noise
  y=noisy-y0
  dat={'scenario':kind,'load':load,'alpha':alpha}
  cross=weighted_cross(J,noise)
  for obs,ids in [('H',IH),('M',IM),('HM',np.arange(90))]:
   r=pdmetric(J,y,wtruth,ids,noise)
   for field in ['info_k','info_E','rmse_wk','rmse_wE','rmse_logk','rmse_logE','coverage90_k','coverage90_E','fro_cross','largest_cross_singular']:
    dat[f'{obs}_{field}']=r[field]
  dat.update(cross)
  rows.append(dat)
  if idx in [0,2,4]:
   detailed[kind]={'J':J, 'wtrue':wtruth, 'condition':dat,
                  'y0':y0,'noisy':noisy,'noise':noise}
  print(idx+1,'of',len(cases),kind,load,'alpha',alpha,
        'cross H,M,HM',*[round(dat[f'{obs}_fro_cross'],3) for obs in ['H','M','HM']],
        'rmse_logk HM',round(dat['HM_rmse_logk'],4),flush=True)
  np.savez_compressed(OUT/f'case_{idx:02d}_{kind}_{load}_a{alpha:.1f}.npz',
     Jw=J,alpha=alpha,baseline_logk=mk0,baseline_logE=me0,
     truth_w=wtruth,noise=noise,observed=noisy,nominal=y0,
     H_indices=IH,M_indices=IM)
 with open(OUT/'summary.csv','w',newline='',encoding='utf-8') as f:
  cols=list(rows[0].keys())
  writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(rows)
 # figure I: normalized info gain in field per scenario obs
 fig,ax=plt.subplots(figsize=(10,5.5))
 labs=[f'{r["scenario"][:4]}:{r["load"][:3]} a={r["alpha"]}' for r in rows]
 x=np.arange(len(rows)); w=.23
 for j,obs in enumerate(['H','M','HM']):
  ax.bar(x+(j-1)*w,[r[f'{obs}_info_k']+r[f'{obs}_info_E'] for r in rows],w,label=obs)
 ax.set_xticks(x,labs,rotation=45,ha='right');ax.set_ylabel('Total Gaussian information gain (trace I-C)'); ax.legend();fig.tight_layout();fig.savefig(OUT/'fig_info_by_observation.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5))
 for obs in ['H','M','HM']:
  ax.plot(np.arange(len(rows)),[r[f'{obs}_fro_cross'] for r in rows],marker='o',label=obs)
 ax.set_xticks(np.arange(len(rows)),labs,rotation=45,ha='right');ax.set_ylabel('||posterior C_(k,E)||_F (white coords)'); ax.legend();fig.tight_layout();fig.savefig(OUT/'fig_cross_covariance.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4))
 ax.plot(np.arange(len(rows)),[r['HM_cross_gram_relative'] if 'HM_cross_gram_relative' in r else r['cross_gram_relative'] for r in rows],marker='o',label='Relative Jk--JE overlap')
 ax.set_xticks(np.arange(len(rows)),labs,rotation=45,ha='right'); ax.set_ylabel('Cross sensitivity overlap');fig.tight_layout();fig.savefig(OUT/'fig_sensitivity_overlap.png',dpi=160);plt.close(fig)
 # selected local posterior response block singular diagnostics
 for kind,r in detailed.items():
  J=r['J']/noise[:,None];co=J[:,:NP].T@J[:,NP:]
  u,s,v=np.linalg.svd(co,full_matrices=False)
  fig,ax=plt.subplots(1,2,figsize=(9,3.5))
  im=ax[0].imshow((T@u[:,0]).reshape(N,N),origin='lower',cmap='coolwarm');ax[0].set_title('log k response direction');fig.colorbar(im,ax=ax[0],shrink=.7)
  im=ax[1].imshow((T@v[0,:]).reshape(N,N),origin='lower',cmap='coolwarm');ax[1].set_title('log E response direction');fig.colorbar(im,ax=ax[1],shrink=.7)
  fig.suptitle(f'Sensitivity cross-Gram largest mode | {kind}')
  fig.tight_layout();fig.savefig(OUT/f'fig_response_direction_{kind}.png',dpi=150);plt.close(fig)
 manifest={'environment':'independent 2D transient quasi-static vector Biot finite difference/energy discretization',
  'not_original_Pilot_solver':True,'unknown_dimension':512,'k_dimension':256,'E_dimension':256,
  'displacement_interior_nodes_per_component':NU,'pressure_interior_nodes':NU,
  'observation_dimensions':{'H':42,'M':48,'HM':90}, 'time_steps':STEPS,
  'alpha_values':[0.,.4,.7,1.], 'nu':nu,'storage':S,'dt':DT,
  'p_u_boundary':'homogeneous Dirichlet','prior':'row-normalized Gaussian spatial smoothing of white prior',
  'physical_parameterization':'log k/E maps from white w via prior map; k/E strictly positive',
  'tangent_validation_relative_errors':errs,'case_count':len(cases),'sigmaH':float(noise[0]),
  'sigmaM':float(noise[14]),'time_sec':time.monotonic()-start,
  'linearization_point':'nominal w=0 for each case',
  'scope':'Gaussian tangent posterior metrics + independent nonlinear forward data for point-estimate errors; NO full nonlinear MCMC reference or Pilot21 Flow repro'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 print('ALL DONE',json.dumps(manifest,ensure_ascii=False),flush=True)
 return rows,manifest

if __name__=='__main__':
 run()