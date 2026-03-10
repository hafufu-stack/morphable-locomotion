"""
Season 5B: 2D Synergy×Env + 3-Body System
==========================================
Exp 14e: 2D Synergy + Friction 0.1:5.0 (can 2D beat 1D's +211?)
Exp 16:  3-Body combination (N-body generalization of Symmetry Locks)
"""

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.stats import pearsonr
import os, time, json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

DT=0.010; GROUND_Y=-0.5; GROUND_K=600.0; GRAVITY=-9.8
BASE_AMP=30.0; DRAG=0.4; SPRING_K=30.0; SPRING_DAMP=1.5; HIDDEN_SIZE=32


# ================================================================
# 2-BODY HELPERS (reused from season5)
# ================================================================
def build_bodies_2(gx, gy, gz, sp, gap):
    nper=gx*gy*gz; nt=nper*2; ap=np.zeros((nt,3)); bi=np.zeros(nt, dtype=np.int64)
    bw=(gx-1)*sp; idx=0
    for b in range(2):
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    xp=(-(gap/2+bw)+x*sp) if b==0 else ((gap/2+bw)-x*sp)
                    ap[idx]=[xp,2.0+y*sp,z*sp-(gz-1)*sp/2]; bi[idx]=b; idx+=1
    sa,sb,rl=[],[],[]
    for b in range(2):
        m=np.where(bi==b)[0]; bp=ap[m]; tri=Delaunay(bp); edges=set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1,4): edges.add((min(m[s[i]],m[s[j]]),max(m[s[i]],m[s[j]])))
        for a,bb in edges: sa.append(a); sb.append(bb); rl.append(np.linalg.norm(ap[a]-ap[bb]))
    np_=np.zeros_like(ap)
    for b in range(2):
        m=bi==b
        for d in range(3):
            vn,vx=ap[m,d].min(),ap[m,d].max(); np_[m,d]=2*(ap[m,d]-vn)/(vx-vn+1e-8)-1
    return ap, np_, bi, np.array(sa), np.array(sb), np.array(rl), nper, nt


@torch.no_grad()
def simulate_2body(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps,
                   input_size, n_syn, syn_str=0.9, fric_map=None, mass_ratios=None):
    OUTPUT_SIZE = 3 + max(n_syn, 0)
    N_W1 = input_size * HIDDEN_SIZE
    B=genomes.shape[0]; pos=rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel=torch.zeros(B,N,3,device=DEVICE)
    idx=0; W1=genomes[:,idx:idx+N_W1].reshape(B,input_size,HIDDEN_SIZE);idx+=N_W1
    b1=genomes[:,idx:idx+HIDDEN_SIZE].unsqueeze(1);idx+=HIDDEN_SIZE
    W2=genomes[:,idx:idx+HIDDEN_SIZE*OUTPUT_SIZE].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);idx+=HIDDEN_SIZE*OUTPUT_SIZE
    b2=genomes[:,idx:idx+OUTPUT_SIZE].unsqueeze(1);idx+=OUTPUT_SIZE
    freq=genomes[:,idx].abs()
    sx=pos[:,:,0].mean(dim=1)
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False; mass=torch.ones(B,N,device=DEVICE)
    if mass_ratios:
        mass[:,b0m] = mass_ratios[0]; mass[:,b1m] = mass_ratios[1]
    # Mode vectors
    mode_vectors=torch.zeros(min(n_syn,50),N,device=DEVICE)
    if n_syn>0: mode_vectors[0,b0m]=1.0;mode_vectors[0,b1m]=-1.0
    if n_syn>1:
        median_y=rp[:,1].median();top=rp[:,1]>=median_y
        mode_vectors[1,top]=1.0;mode_vectors[1,~top]=-1.0
    if n_syn>2:
        median_z=rp[:,2].median();left=rp[:,2]>=median_z
        mode_vectors[2,left]=1.0;mode_vectors[2,~left]=-1.0
    FRIC_DEF=3.0; fric=torch.full((N,),FRIC_DEF,device=DEVICE)
    if fric_map:
        fric[b0m]=fric_map['body0_friction'];fric[b1m]=fric_map['body1_friction']
    base_mass=mass.clone()
    for step in range(nsteps):
        t=step*DT
        if not cd and step%10==0:
            p0=pos[0,b0i];p1=pos[0,b1i];ds=torch.cdist(p0,p1)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,b0i[cl[:nn_,0]]]);csb=torch.cat([csb,b1i[cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                comb=torch.ones(B,1,1,device=DEVICE);cd=True
        st=torch.sin(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        ct=torch.cos(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1);o=torch.tanh(torch.bmm(h,W2)+b2)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        if n_syn>0:
            mass_new=base_mass.clone()
            for d in range(min(n_syn,50)):
                shift=o[:,:,3+d].mean(dim=1)
                mass_new+=syn_str*shift.unsqueeze(1)*mode_vectors[d].unsqueeze(0)
            mass=mass_new.clamp(min=0.1)
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=fric.unsqueeze(0)*vel[:,:,0]*bl;f[:,:,2]-=fric.unsqueeze(0)*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        inv_mass=1.0/mass.clamp(min=0.01);acc=f*inv_mass.unsqueeze(2)
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT

    disp=pos[:,:,0].mean(dim=1)-sx;dz=pos[:,:,2].mean(dim=1).abs()
    sp_=pos.max(dim=1).values-pos.min(dim=1).values
    spp=((sp_-8.0).clamp(min=0)*1.5).sum(dim=1);bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
    c0=pos[:,b0m].mean(dim=1);c1=pos[:,b1m].mean(dim=1)
    coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
    fitness=disp-dz-spp-bw-ep+coh
    fitness=torch.where(torch.isnan(fitness),torch.tensor(-9999.0,device=DEVICE),fitness)
    return fitness, disp


def evolve_2body(data, nsteps, ngens, psz, input_size, n_syn, label,
                 syn_str=0.9, fric_map=None, mass_ratios=None):
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt
    OUTPUT_SIZE=3+max(n_syn,0)
    N_W1=input_size*HIDDEN_SIZE
    N_GENES=N_W1+HIDDEN_SIZE+HIDDEN_SIZE*OUTPUT_SIZE+OUTPUT_SIZE+1
    s1=np.sqrt(2.0/(input_size+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    pop=torch.randn(psz,N_GENES,device=DEVICE)*0.3
    pop[:,:N_W1]*=s1/0.3;pop[:,N_W1:N_W1+HIDDEN_SIZE]=0
    i2=N_W1+HIDDEN_SIZE;pop[:,i2:i2+HIDDEN_SIZE*OUTPUT_SIZE]*=s2/0.3
    pop[:,i2+HIDDEN_SIZE*OUTPUT_SIZE:i2+HIDDEN_SIZE*OUTPUT_SIZE+OUTPUT_SIZE]=0
    pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE)
    gen_log,fitness_log=[],[];t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,_=simulate_2body(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,
                               input_size,n_syn,syn_str,fric_map,mass_ratios)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]
        if gen%50==0 or gen==ngens-1:
            gen_log.append(gen);fitness_log.append(pf[0].item())
            print(f"  [{label}] Gen {gen:4d}/{ngens}: fit={pf[0].item():+.2f} ({(time.time()-t0)/60:.1f}min)")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,N_GENES,device=DEVICE)*0.3
        fr[:,:N_W1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE);p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE);p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,N_GENES,device=DEVICE)<0.5;ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,N_GENES,device=DEVICE)<0.15;ch+=torch.randn(nc,N_GENES,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2
    total=(time.time()-t0)/60;best=pop[0].cpu().numpy()
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return best, gen_log, fitness_log, total


def replay_rfx_2body(genes, data, nsteps, input_size, n_syn, syn_str=0.9,
                     fric_map=None, mass_ratios=None):
    """Replay best genome, return r(Fx) between body0 and body1."""
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUTPUT_SIZE=3+max(n_syn,0)
    N_W1=input_size*HIDDEN_SIZE
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    genome=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    gi=0;W1=genome[:,gi:gi+N_W1].reshape(B,input_size,HIDDEN_SIZE);gi+=N_W1
    b1g=genome[:,gi:gi+HIDDEN_SIZE].unsqueeze(1);gi+=HIDDEN_SIZE
    W2=genome[:,gi:gi+HIDDEN_SIZE*OUTPUT_SIZE].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);gi+=HIDDEN_SIZE*OUTPUT_SIZE
    b2g=genome[:,gi:gi+OUTPUT_SIZE].unsqueeze(1);gi+=OUTPUT_SIZE
    fv=genome[:,gi].abs().item()
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False;mass=torch.ones(B,N,device=DEVICE)
    if mass_ratios: mass[:,b0m]=mass_ratios[0];mass[:,b1m]=mass_ratios[1]
    mode_vectors=torch.zeros(min(n_syn,50),N,device=DEVICE)
    if n_syn>0: mode_vectors[0,b0m]=1.0;mode_vectors[0,b1m]=-1.0
    if n_syn>1:
        median_y=rp[:,1].median();top=rp[:,1]>=median_y
        mode_vectors[1,top]=1.0;mode_vectors[1,~top]=-1.0
    if n_syn>2:
        median_z=rp[:,2].median();left=rp[:,2]>=median_z
        mode_vectors[2,left]=1.0;mode_vectors[2,~left]=-1.0
    FRIC_DEF=3.0;fric=torch.full((N,),FRIC_DEF,device=DEVICE)
    if fric_map: fric[b0m]=fric_map['body0_friction'];fric[b1m]=fric_map['body1_friction']
    base_mass=mass.clone();fx0_l,fx1_l=[],[]
    for step in range(nsteps):
        t=step*DT
        if not cd and step%10==0:
            p0=pos[0,b0i];p1=pos[0,b1i];ds=torch.cdist(p0,p1)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,b0i[cl[:nn_,0]]]);csb=torch.cat([csb,b1i[cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                comb=torch.ones(B,1,1,device=DEVICE);cd=True
        sv=np.sin(2*np.pi*fv*t);cv=np.cos(2*np.pi*fv*t)
        st=torch.full((B,N,1),sv,device=DEVICE);ct=torch.full((B,N,1),cv,device=DEVICE)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        if n_syn>0:
            mass_new=base_mass.clone()
            for d in range(min(n_syn,50)):
                shift=o[:,:,3+d].mean(dim=1)
                mass_new+=syn_str*shift.unsqueeze(1)*mode_vectors[d].unsqueeze(0)
            mass=mass_new.clamp(min=0.1)
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        fx0_l.append(ext[0,b0m,0].mean().item());fx1_l.append(ext[0,b1m,0].mean().item())
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=fric.unsqueeze(0)*vel[:,:,0]*bl;f[:,:,2]-=fric.unsqueeze(0)*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        inv_mass=1.0/mass.clamp(min=0.01);acc=f*inv_mass.unsqueeze(2)
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
    if np.std(fx0_l)<1e-10 or np.std(fx1_l)<1e-10: return 0.0
    r_fx,_=pearsonr(fx0_l,fx1_l);return r_fx


# ================================================================
# 3-BODY SYSTEM
# ================================================================
def build_bodies_3(gx, gy, gz, sp, gap, mass_ratios=(1.0,1.0,1.0)):
    """Build 3 bodies in a line: Body0 (left), Body1 (center), Body2 (right)."""
    nper = gx*gy*gz; nt = nper*3
    ap = np.zeros((nt, 3)); bi = np.zeros(nt, dtype=np.int64)
    bw = (gx-1)*sp
    # Body 0: far left, Body 1: center, Body 2: far right
    offsets = [-(gap+bw), 0.0, (gap+bw)]
    idx = 0
    for b in range(3):
        cx = offsets[b]
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    xp = cx + (x - (gx-1)/2)*sp
                    ap[idx] = [xp, 2.0+y*sp, z*sp-(gz-1)*sp/2]
                    bi[idx] = b; idx += 1
    # Internal springs within each body
    sa, sb, rl = [], [], []
    for b in range(3):
        m = np.where(bi==b)[0]; bp = ap[m]
        tri = Delaunay(bp); edges = set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1, 4):
                    edges.add((min(m[s[i]],m[s[j]]), max(m[s[i]],m[s[j]])))
        for a, bb in edges:
            sa.append(a); sb.append(bb); rl.append(np.linalg.norm(ap[a]-ap[bb]))
    # Normalized positions within each body
    np_ = np.zeros_like(ap)
    for b in range(3):
        m = bi == b
        for d in range(3):
            vn, vx = ap[m,d].min(), ap[m,d].max()
            np_[m,d] = 2*(ap[m,d]-vn)/(vx-vn+1e-8)-1
    return ap, np_, bi, np.array(sa), np.array(sb), np.array(rl), nper, nt


@torch.no_grad()
def simulate_3body(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps,
                   input_size, mass_ratios=(1.0,1.0,1.0)):
    """3-body simulation. body_id is encoded as 0.0, 0.5, 1.0 for bodies 0,1,2."""
    OUTPUT_SIZE = 3
    N_W1 = input_size * HIDDEN_SIZE
    B = genomes.shape[0]; pos = rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel = torch.zeros(B,N,3,device=DEVICE)
    idx=0; W1=genomes[:,idx:idx+N_W1].reshape(B,input_size,HIDDEN_SIZE);idx+=N_W1
    b1=genomes[:,idx:idx+HIDDEN_SIZE].unsqueeze(1);idx+=HIDDEN_SIZE
    W2=genomes[:,idx:idx+HIDDEN_SIZE*OUTPUT_SIZE].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);idx+=HIDDEN_SIZE*OUTPUT_SIZE
    b2=genomes[:,idx:idx+OUTPUT_SIZE].unsqueeze(1);idx+=OUTPUT_SIZE
    freq=genomes[:,idx].abs()
    sx = pos[:,:,0].mean(dim=1)
    # body_id: 0→0.0, 1→0.5, 2→1.0 (normalized)
    bid_vals = bit.float() / 2.0  # 0, 0.5, 1.0
    bid = bid_vals.unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni = npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone(); csb=sbt.clone(); crl=rlt.clone()
    # Combination tracking: 0→1, 1→2 (sequential combination)
    comb01 = torch.zeros(B,1,1,device=DEVICE)
    comb12 = torch.zeros(B,1,1,device=DEVICE)
    te = torch.zeros(B,device=DEVICE)
    masks = [(bit==b) for b in range(3)]
    idxs = [masks[b].nonzero(as_tuple=True)[0] for b in range(3)]
    cd01 = False; cd12 = False
    mass = torch.ones(B,N,device=DEVICE)
    for b in range(3):
        mass[:, masks[b]] = mass_ratios[b]
    FRICTION = 3.0

    for step in range(nsteps):
        t = step * DT
        # Check combination Body0-Body1
        if not cd01 and step%10==0:
            p0=pos[0,idxs[0]]; p1=pos[0,idxs[1]]; ds=torch.cdist(p0,p1)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,idxs[0][cl[:nn_,0]]]);csb=torch.cat([csb,idxs[1][cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                comb01=torch.ones(B,1,1,device=DEVICE); cd01=True
        # Check combination Body1-Body2
        if not cd12 and step%10==0:
            p1=pos[0,idxs[1]]; p2=pos[0,idxs[2]]; ds=torch.cdist(p1,p2)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,idxs[1][cl[:nn_,0]]]);csb=torch.cat([csb,idxs[2][cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                comb12=torch.ones(B,1,1,device=DEVICE); cd12=True
        # Combined status: max of both
        comb_any = torch.max(comb01, comb12)
        st=torch.sin(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        ct=torch.cos(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        nn_in=torch.cat([st,ct,ni,bid,comb_any.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1); o=torch.tanh(torch.bmm(h,W2)+b2)
        og=(pos[:,:,1]<GROUND_Y+0.3).float(); gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc
        ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        te+=(ext**2).sum(dim=(1,2))
        f=torch.zeros(B,N,3,device=DEVICE); f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0); f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=FRICTION*vel[:,:,0]*bl; f[:,:,2]-=FRICTION*vel[:,:,2]*bl
        f-=DRAG*vel; f+=ext
        inv_mass=1.0/mass.clamp(min=0.01); acc=f*inv_mass.unsqueeze(2)
        vel+=acc*DT; vel.clamp_(-50,50); pos+=vel*DT

    disp=pos[:,:,0].mean(dim=1)-sx
    dz=pos[:,:,2].mean(dim=1).abs()
    sp_=pos.max(dim=1).values-pos.min(dim=1).values
    spp=((sp_-10.0).clamp(min=0)*1.5).sum(dim=1)  # bigger spread limit for 3 bodies
    bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3; ep=1.0*(te/me)*100
    # Cohesion: all 3 body centers should stay close
    c=[pos[:,masks[b]].mean(dim=1) for b in range(3)]
    d01=torch.norm(c[0]-c[1],dim=1); d12=torch.norm(c[1]-c[2],dim=1)
    coh=torch.clamp(3.0-d01,min=0)*1.5 + torch.clamp(3.0-d12,min=0)*1.5
    fitness=disp-dz-spp-bw-ep+coh
    fitness=torch.where(torch.isnan(fitness),torch.tensor(-9999.0,device=DEVICE),fitness)
    return fitness, disp


def evolve_3body(data, nsteps, ngens, psz, input_size, label, mass_ratios=(1.0,1.0,1.0)):
    ap,np_,bi,sa,sb,rl,nper,nt=data; N=nt; OUTPUT_SIZE=3
    N_W1=input_size*HIDDEN_SIZE
    N_GENES=N_W1+HIDDEN_SIZE+HIDDEN_SIZE*OUTPUT_SIZE+OUTPUT_SIZE+1
    s1=np.sqrt(2.0/(input_size+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    pop=torch.randn(psz,N_GENES,device=DEVICE)*0.3
    pop[:,:N_W1]*=s1/0.3;pop[:,N_W1:N_W1+HIDDEN_SIZE]=0
    i2=N_W1+HIDDEN_SIZE;pop[:,i2:i2+HIDDEN_SIZE*OUTPUT_SIZE]*=s2/0.3
    pop[:,i2+HIDDEN_SIZE*OUTPUT_SIZE:i2+HIDDEN_SIZE*OUTPUT_SIZE+OUTPUT_SIZE]=0
    pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE)
    gen_log,fitness_log=[],[];t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,_=simulate_3body(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,
                               input_size,mass_ratios)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]
        if gen%50==0 or gen==ngens-1:
            gen_log.append(gen);fitness_log.append(pf[0].item())
            print(f"  [{label}] Gen {gen:4d}/{ngens}: fit={pf[0].item():+.2f} ({(time.time()-t0)/60:.1f}min)")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,N_GENES,device=DEVICE)*0.3
        fr[:,:N_W1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE);p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE);p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,N_GENES,device=DEVICE)<0.5;ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,N_GENES,device=DEVICE)<0.15;ch+=torch.randn(nc,N_GENES,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2
    total=(time.time()-t0)/60;best=pop[0].cpu().numpy()
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return best, gen_log, fitness_log, total


def replay_rfx_3body(genes, data, nsteps, input_size, mass_ratios=(1.0,1.0,1.0)):
    """Replay best genome, return pairwise r(Fx) for 3 bodies."""
    ap,np_,bi,sa,sb,rl,nper,nt=data;N=nt;OUTPUT_SIZE=3
    N_W1=input_size*HIDDEN_SIZE
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    genome=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    gi=0;W1=genome[:,gi:gi+N_W1].reshape(B,input_size,HIDDEN_SIZE);gi+=N_W1
    b1g=genome[:,gi:gi+HIDDEN_SIZE].unsqueeze(1);gi+=HIDDEN_SIZE
    W2=genome[:,gi:gi+HIDDEN_SIZE*OUTPUT_SIZE].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);gi+=HIDDEN_SIZE*OUTPUT_SIZE
    b2g=genome[:,gi:gi+OUTPUT_SIZE].unsqueeze(1);gi+=OUTPUT_SIZE
    fv=genome[:,gi].abs().item()
    bid_vals=bit.float()/2.0;bid=bid_vals.unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb01=torch.zeros(B,1,1,device=DEVICE);comb12=torch.zeros(B,1,1,device=DEVICE)
    masks=[(bit==b) for b in range(3)];idxs=[masks[b].nonzero(as_tuple=True)[0] for b in range(3)]
    cd01=False;cd12=False
    mass=torch.ones(B,N,device=DEVICE)
    for b in range(3): mass[:,masks[b]]=mass_ratios[b]
    FRICTION=3.0
    fx = {0:[],1:[],2:[]}
    for step in range(nsteps):
        t=step*DT
        if not cd01 and step%10==0:
            p0=pos[0,idxs[0]];p1=pos[0,idxs[1]];ds=torch.cdist(p0,p1)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,idxs[0][cl[:nn_,0]]]);csb=torch.cat([csb,idxs[1][cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                comb01=torch.ones(B,1,1,device=DEVICE);cd01=True
        if not cd12 and step%10==0:
            p1=pos[0,idxs[1]];p2=pos[0,idxs[2]];ds=torch.cdist(p1,p2)
            cl=(ds<1.2).nonzero(as_tuple=False)
            if cl.shape[0]>0:
                nn_=min(cl.shape[0],500)
                csa=torch.cat([csa,idxs[1][cl[:nn_,0]]]);csb=torch.cat([csb,idxs[2][cl[:nn_,1]]])
                crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                comb12=torch.ones(B,1,1,device=DEVICE);cd12=True
        comb_any=torch.max(comb01,comb12)
        sv=np.sin(2*np.pi*fv*t);cv=np.cos(2*np.pi*fv*t)
        st=torch.full((B,N,1),sv,device=DEVICE);ct=torch.full((B,N,1),cv,device=DEVICE)
        nn_in=torch.cat([st,ct,ni,bid,comb_any.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        for b in range(3): fx[b].append(ext[0,masks[b],0].mean().item())
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d_=pb-pa;di=torch.norm(d_,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d_/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft_=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft_)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft_)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=FRICTION*vel[:,:,0]*bl;f[:,:,2]-=FRICTION*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        inv_mass=1.0/mass.clamp(min=0.01);acc=f*inv_mass.unsqueeze(2)
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
    # Pairwise correlations
    pairs = {}
    for (a,b) in [(0,1),(1,2),(0,2)]:
        if np.std(fx[a])<1e-10 or np.std(fx[b])<1e-10:
            pairs[f"r{a}{b}"] = 0.0
        else:
            r,_ = pearsonr(fx[a], fx[b])
            pairs[f"r{a}{b}"] = round(r, 3)
    return pairs


def main():
    NSTEPS=600; GAP=0.5; PSZ=200; INPUT_SIZE=7
    gx,gy,gz,sp=10,5,4,0.35
    results = {}

    # ================================================================
    print("="*70)
    print("EXP 14e: 2D SYNERGY × ENVIRONMENT (Can 2D beat 1D's +211?)")
    print("="*70)
    data2 = build_bodies_2(gx,gy,gz,sp,GAP)
    fmap = {"body0_friction": 0.1, "body1_friction": 5.0}

    bg,gl,fl,elapsed = evolve_2body(data2,NSTEPS,300,PSZ,INPUT_SIZE,2,"14e_2D_syn_env",
                                    fric_map=fmap)
    rfx = replay_rfx_2body(bg,data2,NSTEPS,INPUT_SIZE,2,fric_map=fmap)
    results["exp14e"] = {"label":"2D_synergy_env","fitness":round(fl[-1],2),"r_fx":round(rfx,3)}
    print(f"  >>> 2D Synergy×Env: fit={fl[-1]:+.2f}, r(Fx)={rfx:.3f}")
    print(f"  (vs 1D Synergy×Env: +211.42, vs 2D Synergy sym: +216.41)")

    # ================================================================
    print("\n" + "="*70)
    print("EXP 16: 3-BODY COMBINATION (N-body Symmetry Locks)")
    print("="*70)
    NGENS_16 = 300
    exp16_results = []
    data3 = build_bodies_3(gx,gy,gz,sp,GAP)
    print(f"  3-body system: {data3[7]} particles ({data3[6]} per body)")

    configs_16 = [
        # (mass_ratios, label)
        ((1.0, 1.0, 1.0), "16a_sym_111"),
        ((3.0, 1.0, 1.0), "16b_asym_311"),
        ((3.0, 2.0, 1.0), "16c_grad_321"),
        ((5.0, 1.0, 5.0), "16d_mirror_515"),
    ]

    for mrs, label in configs_16:
        print(f"\n--- {label}: mass={mrs} ---")
        data3_m = build_bodies_3(gx,gy,gz,sp,GAP,mrs)
        bg,gl,fl,elapsed = evolve_3body(data3_m,NSTEPS,NGENS_16,PSZ,INPUT_SIZE,label,mrs)
        rfx = replay_rfx_3body(bg,data3_m,NSTEPS,INPUT_SIZE,mrs)
        exp16_results.append({
            "label":label,"mass":f"{mrs[0]:.0f}:{mrs[1]:.0f}:{mrs[2]:.0f}",
            "fitness":round(fl[-1],2),"r01":rfx["r01"],"r12":rfx["r12"],"r02":rfx["r02"],
            "elapsed":round(elapsed,1),"gen_log":gl,"fitness_log":fl
        })
        print(f"  fit={fl[-1]:+.2f} | r01={rfx['r01']:.3f} r12={rfx['r12']:.3f} r02={rfx['r02']:.3f}")

    results["exp16_3body"] = exp16_results

    # ================================================================
    # FIGURE
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Season 5B: 2D Synergy×Env & 3-Body Combination", fontsize=14, fontweight="bold")

    # Panel 1: Exp 14e comparison
    ax = axes[0]
    compare_data = [
        ("Fixed\nsym", 170.68, "#95a5a6"),
        ("Fixed\nenv", 183.49, "#2ecc71"),
        ("1D Syn\nsym", 213.76, "#9b59b6"),
        ("1D Syn\nenv", 211.42, "#e67e22"),
        ("2D Syn\nsym", 216.41, "#8e44ad"),
        ("2D Syn\nenv", results["exp14e"]["fitness"], "#e74c3c"),
    ]
    labels_c = [d[0] for d in compare_data]
    fits_c = [d[1] for d in compare_data]
    colors_c = [d[2] for d in compare_data]
    bars = ax.bar(range(len(compare_data)), fits_c, color=colors_c, alpha=0.85)
    for i,d in enumerate(compare_data):
        ax.text(i, fits_c[i]+2, f"{fits_c[i]:+.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(compare_data)));ax.set_xticklabels(labels_c,fontsize=7)
    ax.set_ylabel("Fitness");ax.set_title("Exp 14e: 2D Synergy × Environment")
    ax.axhline(y=216.41,color="purple",linestyle="--",alpha=0.3,label="2D sym record +216")
    ax.legend(fontsize=7);ax.grid(alpha=0.3,axis="y")

    # Panel 2: 3-body r(Fx) heatmap-style
    ax = axes[1]
    n = len(exp16_results)
    x = np.arange(n)
    w = 0.25
    fits = [r["fitness"] for r in exp16_results]
    r01s = [r["r01"] for r in exp16_results]
    r12s = [r["r12"] for r in exp16_results]
    r02s = [r["r02"] for r in exp16_results]
    ax.bar(x-w, fits, w*2, color="#e74c3c", alpha=0.7, label="Fitness")
    ax_r = ax.twinx()
    ax_r.plot(x, r01s, 'o-', color="#3498db", linewidth=2, label="r(0-1)")
    ax_r.plot(x, r12s, 's--', color="#2ecc71", linewidth=2, label="r(1-2)")
    ax_r.plot(x, r02s, '^:', color="#f39c12", linewidth=2, label="r(0-2)")
    mass_labels = [r["mass"] for r in exp16_results]
    ax.set_xticks(x);ax.set_xticklabels(mass_labels,fontsize=9)
    ax.set_xlabel("Mass Ratio (Body0:Body1:Body2)")
    ax.set_ylabel("Fitness",color="#e74c3c")
    ax_r.set_ylabel("r(Fx) pairwise",color="#3498db")
    ax.set_title("Exp 16: 3-Body Combination\n(Symmetry Locks: N-body Generalization)")
    lines1,labels1=ax.get_legend_handles_labels()
    lines2,labels2=ax_r.get_legend_handles_labels()
    ax.legend(lines1+lines2,labels1+labels2,fontsize=7,loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "season5b_experiments.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")

    log_path = os.path.join(RESULTS_DIR, "season5b_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Log saved: {log_path}")

    # Final summary
    print("\n" + "="*70)
    print("SEASON 5B SUMMARY")
    print("="*70)
    print(f"\nExp 14e: 2D Synergy×Env = {results['exp14e']['fitness']:+.2f} (r={results['exp14e']['r_fx']:.3f})")
    print(f"  vs 1D Synergy×Env = +211.42 | vs 2D Synergy sym = +216.41")

    print(f"\nExp 16: 3-Body System")
    for r in exp16_results:
        print(f"  {r['mass']:>7s}  fit={r['fitness']:+8.2f}  "
              f"r01={r['r01']:+.3f}  r12={r['r12']:+.3f}  r02={r['r02']:+.3f}")
    # Interpret
    sym = exp16_results[0]
    if sym["r01"] > 0.6 and sym["r12"] > 0.6:
        print(f"\n  ✅ Symmetry Locks confirmed for 3-body (all r>0.6)")
    asym = [r for r in exp16_results if r["mass"] != "1:1:1"]
    for r in asym:
        if min(r["r01"], r["r12"], r["r02"]) < 0.4:
            print(f"  🔥 Differentiation detected in {r['mass']} (min r < 0.4)")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800,300); time.sleep(0.2)
    except: pass
    print("\nAll Season 5B experiments complete!")


if __name__ == "__main__":
    main()
