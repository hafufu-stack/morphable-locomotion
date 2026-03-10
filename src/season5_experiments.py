"""
Season 5: Synergy × Environment + Synergy Dimension Sweet Spot
===============================================================
Exp 14: Muscle Synergy (1D shift α=0.9) + Friction asymmetry (0.1:5.0)
       → Does dynamic asymmetry bypass the Interference Principle?
Exp 15: Synergy dimension sweep (1D/2D/3D/5D/10D/50D)
       → Where is the critical DoF Trap dimension?
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
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

DT = 0.010; GROUND_Y = -0.5; GROUND_K = 600.0; GRAVITY = -9.8
BASE_AMP = 30.0; DRAG = 0.4; SPRING_K = 30.0; SPRING_DAMP = 1.5
HIDDEN_SIZE = 32


def build_bodies(gx, gy, gz, sp, gap):
    nper = gx*gy*gz; nt = nper*2; ap = np.zeros((nt,3)); bi = np.zeros(nt, dtype=np.int64)
    bw = (gx-1)*sp; idx = 0
    for b in range(2):
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    xp = (-(gap/2+bw)+x*sp) if b==0 else ((gap/2+bw)-x*sp)
                    ap[idx] = [xp, 2.0+y*sp, z*sp-(gz-1)*sp/2]; bi[idx] = b; idx += 1
    sa, sb, rl = [], [], []
    for b in range(2):
        m = np.where(bi==b)[0]; bp = ap[m]; tri = Delaunay(bp); edges = set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1,4): edges.add((min(m[s[i]],m[s[j]]),max(m[s[i]],m[s[j]])))
        for a,bb in edges: sa.append(a);sb.append(bb);rl.append(np.linalg.norm(ap[a]-ap[bb]))
    np_ = np.zeros_like(ap)
    for b in range(2):
        m = bi==b
        for d in range(3):
            vn,vx = ap[m,d].min(),ap[m,d].max(); np_[m,d] = 2*(ap[m,d]-vn)/(vx-vn+1e-8)-1
    return ap, np_, bi, np.array(sa), np.array(sb), np.array(rl), nper, nt


@torch.no_grad()
def simulate_nd_synergy(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps,
                        input_size, n_synergy_dims, synergy_strength=0.9,
                        friction_map=None):
    """
    Generalized N-dimensional synergy simulation.
    n_synergy_dims: number of synergy dimensions (0=fixed, 1=1D shift, 2=2D, etc.)
    For each synergy dim, the NN outputs one extra value averaged across particles.
    The synergy dims shift mass along different spatial modes:
      dim 0: body0 vs body1 (front-back)
      dim 1: top vs bottom (within each body)
      dim 2: left vs right (within each body)
      dim 3+: random orthogonal modes
    """
    OUTPUT_SIZE = 3 + max(n_synergy_dims, 0)
    N_W1 = input_size * HIDDEN_SIZE
    FRICTION_DEFAULT = 3.0

    B = genomes.shape[0]; pos = rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel = torch.zeros(B,N,3,device=DEVICE)
    idx=0; W1=genomes[:,idx:idx+N_W1].reshape(B,input_size,HIDDEN_SIZE);idx+=N_W1
    b1=genomes[:,idx:idx+HIDDEN_SIZE].unsqueeze(1);idx+=HIDDEN_SIZE
    W2=genomes[:,idx:idx+HIDDEN_SIZE*OUTPUT_SIZE].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);idx+=HIDDEN_SIZE*OUTPUT_SIZE
    b2=genomes[:,idx:idx+OUTPUT_SIZE].unsqueeze(1);idx+=OUTPUT_SIZE
    freq=genomes[:,idx].abs()
    sx=pos[:,:,0].mean(dim=1);bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False; mass = torch.ones(B,N,device=DEVICE)

    # Pre-compute synergy mode vectors (N-dimensional)
    # Mode 0: body0=+1, body1=-1 (inter-body shift)
    # Mode 1: top half=+1, bottom half=-1 (vertical, within each body)
    # Mode 2: left=+1, right=-1 (lateral, within each body)
    # Mode 3+: random orthogonal
    mode_vectors = torch.zeros(min(n_synergy_dims, 50), N, device=DEVICE)
    if n_synergy_dims > 0:
        mode_vectors[0, b0m] = 1.0; mode_vectors[0, b1m] = -1.0
    if n_synergy_dims > 1:
        # Top vs bottom (by y-coordinate at init)
        median_y = rp[:, 1].median()
        top_mask = rp[:, 1] >= median_y
        mode_vectors[1, top_mask] = 1.0; mode_vectors[1, ~top_mask] = -1.0
    if n_synergy_dims > 2:
        # Left vs right (by z-coordinate at init)
        median_z = rp[:, 2].median()
        left_mask = rp[:, 2] >= median_z
        mode_vectors[2, left_mask] = 1.0; mode_vectors[2, ~left_mask] = -1.0
    if n_synergy_dims > 3:
        # Random orthogonal modes for dims 3+
        torch.manual_seed(42)
        for d in range(3, min(n_synergy_dims, 50)):
            rv = torch.randn(N, device=DEVICE)
            # Gram-Schmidt against previous modes
            for prev in range(d):
                rv -= (rv * mode_vectors[prev]).sum() / (mode_vectors[prev] ** 2).sum() * mode_vectors[prev]
            rv = rv / (rv.norm() + 1e-8) * np.sqrt(N)
            mode_vectors[d] = rv

    # Friction
    fric = torch.full((N,), FRICTION_DEFAULT, device=DEVICE)
    if friction_map:
        fric[b0m] = friction_map['body0_friction']
        fric[b1m] = friction_map['body1_friction']

    for step in range(nsteps):
        t = step * DT
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

        # N-D Synergy mass control
        if n_synergy_dims > 0:
            mass_new = torch.ones(B,N,device=DEVICE)
            for d in range(min(n_synergy_dims, 50)):
                shift = o[:,:,3+d].mean(dim=1)  # (B,) scalar per individual
                # Apply mode vector: mass += strength * shift * mode_vector
                mass_new += synergy_strength * shift.unsqueeze(1) * mode_vectors[d].unsqueeze(0)
            mass = mass_new.clamp(min=0.1)

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
        f[:,:,0]-=fric.unsqueeze(0)*vel[:,:,0]*bl
        f[:,:,2]-=fric.unsqueeze(0)*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        inv_mass = 1.0 / mass.clamp(min=0.01)
        acc = f * inv_mass.unsqueeze(2)
        vel+=acc*DT; vel.clamp_(-50, 50); pos+=vel*DT

    disp=pos[:,:,0].mean(dim=1)-sx;dz=pos[:,:,2].mean(dim=1).abs()
    sp_=pos.max(dim=1).values-pos.min(dim=1).values
    spp=((sp_-8.0).clamp(min=0)*1.5).sum(dim=1);bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
    c0=pos[:,b0m].mean(dim=1);c1=pos[:,b1m].mean(dim=1)
    coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
    fitness = disp-dz-spp-bw-ep+coh
    fitness = torch.where(torch.isnan(fitness), torch.tensor(-9999.0, device=DEVICE), fitness)
    return fitness, disp


def replay_rfx(genes, data, nsteps, input_size, n_synergy_dims, synergy_strength=0.9,
               friction_map=None):
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt
    OUTPUT_SIZE = 3 + max(n_synergy_dims, 0)
    N_W1 = input_size * HIDDEN_SIZE
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    genome=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    gidx=0;W1=genome[:,gidx:gidx+N_W1].reshape(B,input_size,HIDDEN_SIZE);gidx+=N_W1
    b1g=genome[:,gidx:gidx+HIDDEN_SIZE].unsqueeze(1);gidx+=HIDDEN_SIZE
    W2=genome[:,gidx:gidx+HIDDEN_SIZE*OUTPUT_SIZE].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);gidx+=HIDDEN_SIZE*OUTPUT_SIZE
    b2g=genome[:,gidx:gidx+OUTPUT_SIZE].unsqueeze(1);gidx+=OUTPUT_SIZE
    freq_val=genome[:,gidx].abs().item()
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False;mass=torch.ones(B,N,device=DEVICE)
    FRICTION_DEFAULT=3.0
    fric=torch.full((N,),FRICTION_DEFAULT,device=DEVICE)
    if friction_map:
        fric[b0m]=friction_map['body0_friction'];fric[b1m]=friction_map['body1_friction']

    # Mode vectors (same as simulate)
    mode_vectors=torch.zeros(min(n_synergy_dims,50),N,device=DEVICE)
    if n_synergy_dims>0: mode_vectors[0,b0m]=1.0;mode_vectors[0,b1m]=-1.0
    if n_synergy_dims>1:
        median_y=rp[:,1].median();top_mask=rp[:,1]>=median_y
        mode_vectors[1,top_mask]=1.0;mode_vectors[1,~top_mask]=-1.0
    if n_synergy_dims>2:
        median_z=rp[:,2].median();left_mask=rp[:,2]>=median_z
        mode_vectors[2,left_mask]=1.0;mode_vectors[2,~left_mask]=-1.0
    if n_synergy_dims>3:
        torch.manual_seed(42)
        for d in range(3,min(n_synergy_dims,50)):
            rv=torch.randn(N,device=DEVICE)
            for prev in range(d):
                rv-=(rv*mode_vectors[prev]).sum()/(mode_vectors[prev]**2).sum()*mode_vectors[prev]
            rv=rv/(rv.norm()+1e-8)*np.sqrt(N);mode_vectors[d]=rv

    fx0_list,fx1_list=[],[]
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
        sin_val=np.sin(2*np.pi*freq_val*t);cos_val=np.cos(2*np.pi*freq_val*t)
        st=torch.full((B,N,1),sin_val,device=DEVICE);ct=torch.full((B,N,1),cos_val,device=DEVICE)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        if n_synergy_dims>0:
            mass_new=torch.ones(B,N,device=DEVICE)
            for d in range(min(n_synergy_dims,50)):
                shift=o[:,:,3+d].mean(dim=1)
                mass_new+=synergy_strength*shift.unsqueeze(1)*mode_vectors[d].unsqueeze(0)
            mass=mass_new.clamp(min=0.1)
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        fx0_list.append(ext[0,b0m,0].mean().item());fx1_list.append(ext[0,b1m,0].mean().item())
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
    if np.std(fx0_list)<1e-10 or np.std(fx1_list)<1e-10: return 0.0
    r_fx,_=pearsonr(fx0_list,fx1_list)
    return r_fx


def evolve(data, nsteps, ngens, psz, input_size, n_synergy_dims, label,
           synergy_strength=0.9, friction_map=None):
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt
    OUTPUT_SIZE = 3 + max(n_synergy_dims, 0)
    N_W1 = input_size * HIDDEN_SIZE
    N_GENES = input_size*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE + 1
    s1=np.sqrt(2.0/(input_size+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    pop=torch.randn(psz,N_GENES,device=DEVICE)*0.3
    pop[:,:N_W1]*=s1/0.3;pop[:,N_W1:N_W1+HIDDEN_SIZE]=0
    pop[:,N_W1+HIDDEN_SIZE:N_W1+HIDDEN_SIZE+HIDDEN_SIZE*OUTPUT_SIZE]*=s2/0.3
    pop[:,N_W1+HIDDEN_SIZE+HIDDEN_SIZE*OUTPUT_SIZE:N_W1+HIDDEN_SIZE+HIDDEN_SIZE*OUTPUT_SIZE+OUTPUT_SIZE]=0
    pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE)
    gen_log,fitness_log=[],[];t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,_=simulate_nd_synergy(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,
                                    input_size,n_synergy_dims,synergy_strength,friction_map)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]
        if gen%50==0 or gen==ngens-1:
            gen_log.append(gen);fitness_log.append(pf[0].item())
            elapsed=time.time()-t0
            print(f"  [{label}] Gen {gen:4d}/{ngens}: fit={pf[0].item():+.2f}  ({elapsed/60:.1f}min)")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,N_GENES,device=DEVICE)*0.3
        fr[:,:N_W1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE)
        p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE)
        p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,N_GENES,device=DEVICE)<0.5
        ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,N_GENES,device=DEVICE)<0.15
        ch+=torch.randn(nc,N_GENES,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2
    total=(time.time()-t0)/60;best_genes=pop[0].cpu().numpy()
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return best_genes, gen_log, fitness_log, total


def main():
    NSTEPS=600; GAP=0.5; PSZ=200; INPUT_SIZE=7
    gx,gy,gz,sp=10,5,4,0.35
    data=build_bodies(gx,gy,gz,sp,GAP)
    results={}

    # ================================================================
    print("="*70)
    print("EXP 14: SYNERGY × ENVIRONMENT (Interference or Resonance?)")
    print("="*70)
    NGENS_14 = 300
    fmap = {"body0_friction": 0.1, "body1_friction": 5.0}
    exp14_results = []

    # 14a: Synergy 1D, symmetric friction (reproduce +210 baseline)
    print("\n--- 14a: Synergy 1D, symmetric friction (control) ---")
    bg,gl,fl,elapsed = evolve(data,NSTEPS,NGENS_14,PSZ,INPUT_SIZE,1,"14a_syn_sym")
    rfx = replay_rfx(bg,data,NSTEPS,INPUT_SIZE,1)
    exp14_results.append({"label":"synergy_sym","n_syn":1,"friction":"sym","fitness":round(fl[-1],2),"r_fx":round(rfx,3)})
    print(f"  r(Fx)={rfx:.3f}, fit={fl[-1]:+.2f}")

    # 14b: Synergy 1D + friction asymmetry 0.1:5.0 → RESONANCE or INTERFERENCE?
    print("\n--- 14b: Synergy 1D + friction 0.1:5.0 (THE test) ---")
    bg,gl,fl,elapsed = evolve(data,NSTEPS,NGENS_14,PSZ,INPUT_SIZE,1,"14b_syn_env",
                              friction_map=fmap)
    rfx = replay_rfx(bg,data,NSTEPS,INPUT_SIZE,1,friction_map=fmap)
    exp14_results.append({"label":"synergy_env","n_syn":1,"friction":"0.1:5","fitness":round(fl[-1],2),"r_fx":round(rfx,3)})
    print(f"  r(Fx)={rfx:.3f}, fit={fl[-1]:+.2f}")

    # 14c: Fixed 3-out + friction asymmetry (Exp 9 reproduction for comparison)
    print("\n--- 14c: Fixed 3-out + friction 0.1:5.0 (Exp 9 control) ---")
    bg,gl,fl,elapsed = evolve(data,NSTEPS,150,PSZ,INPUT_SIZE,0,"14c_fixed_env",
                              friction_map=fmap)
    rfx = replay_rfx(bg,data,NSTEPS,INPUT_SIZE,0,friction_map=fmap)
    exp14_results.append({"label":"fixed_env","n_syn":0,"friction":"0.1:5","fitness":round(fl[-1],2),"r_fx":round(rfx,3)})
    print(f"  r(Fx)={rfx:.3f}, fit={fl[-1]:+.2f}")

    # 14d: Fixed 3-out, symmetric (baseline)
    print("\n--- 14d: Fixed 3-out, symmetric (baseline) ---")
    bg,gl,fl,elapsed = evolve(data,NSTEPS,150,PSZ,INPUT_SIZE,0,"14d_fixed_sym")
    rfx = replay_rfx(bg,data,NSTEPS,INPUT_SIZE,0)
    exp14_results.append({"label":"fixed_sym","n_syn":0,"friction":"sym","fitness":round(fl[-1],2),"r_fx":round(rfx,3)})
    print(f"  r(Fx)={rfx:.3f}, fit={fl[-1]:+.2f}")

    results["exp14_synergy_env"] = exp14_results

    # ================================================================
    print("\n" + "="*70)
    print("EXP 15: SYNERGY DIMENSION SWEET SPOT (Critical DoF)")
    print("="*70)
    NGENS_15 = 300
    exp15_results = []

    dims_to_test = [0, 1, 2, 3, 5, 10, 50]

    for nd in dims_to_test:
        label = f"15_{nd}D"
        n_out = 3 + nd
        print(f"\n--- {label}: {nd}D synergy ({n_out} outputs, {NGENS_15}gen) ---")
        bg,gl,fl,elapsed = evolve(data,NSTEPS,NGENS_15,PSZ,INPUT_SIZE,nd,label)
        rfx = replay_rfx(bg,data,NSTEPS,INPUT_SIZE,nd)
        exp15_results.append({
            "label":f"{nd}D","n_dims":nd,"n_outputs":n_out,
            "fitness":round(fl[-1],2),"r_fx":round(rfx,3),"elapsed":round(elapsed,1)
        })
        print(f"  r(Fx)={rfx:.3f}, fit={fl[-1]:+.2f}")

    results["exp15_synergy_dims"] = exp15_results

    # ================================================================
    # SUMMARY FIGURE
    # ================================================================
    print("\nGenerating summary figure...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Season 5: Synergy × Environment & Dimension Sweet Spot",
                 fontsize=14, fontweight="bold")

    # Panel 1: Exp 14
    ax = axes[0]
    labels_14 = [r["label"].replace("_","\n") for r in exp14_results]
    fits_14 = [r["fitness"] for r in exp14_results]
    colors_14 = ["#9b59b6","#e74c3c","#2ecc71","#95a5a6"]
    bars = ax.bar(range(len(exp14_results)), fits_14, color=colors_14, alpha=0.8)
    for i,r in enumerate(exp14_results):
        y_pos = max(fits_14[i]+3, 5)
        ax.text(i, y_pos, f"fit={fits_14[i]:+.0f}\nr={r['r_fx']:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(exp14_results)))
    ax.set_xticklabels(labels_14, fontsize=7)
    ax.set_ylabel("Fitness")
    ax.set_title("Exp 14: Synergy × Environment\n(Interference or Resonance?)")
    # Reference lines
    ax.axhline(y=210, color="purple", linestyle="--", alpha=0.3, label="Synergy record (+210)")
    ax.axhline(y=179, color="green", linestyle="--", alpha=0.3, label="Env only record (+179)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    # Panel 2: Exp 15
    ax = axes[1]
    dims = [r["n_dims"] for r in exp15_results]
    fits_15 = [r["fitness"] for r in exp15_results]
    rfx_15 = [r["r_fx"] for r in exp15_results]
    color_map = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(dims)))

    ax2 = ax.twinx()
    ax.plot(range(len(dims)), fits_15, 'o-', color="#e74c3c", linewidth=2, markersize=8, label="Fitness", zorder=5)
    ax2.plot(range(len(dims)), rfx_15, 's--', color="#3498db", linewidth=2, markersize=6, label="r(Fx)", alpha=0.7)
    for i,r in enumerate(exp15_results):
        ax.annotate(f"{fits_15[i]:+.0f}", (i, fits_15[i]), textcoords="offset points",
                    xytext=(0,12), ha="center", fontsize=8, fontweight="bold", color="#e74c3c")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"{d}D\n({3+d}out)" for d in dims], fontsize=7)
    ax.set_ylabel("Fitness", color="#e74c3c")
    ax2.set_ylabel("r(Fx)", color="#3498db")
    ax.set_xlabel("Synergy Dimensions")
    ax.set_title("Exp 15: Synergy Dimension Sweet Spot\n(Critical DoF for DoF Trap)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax2.axhline(y=0.3, color="blue", linestyle="--", alpha=0.3, label="Diff threshold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "season5_experiments.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")

    log_path = os.path.join(RESULTS_DIR, "season5_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Log saved: {log_path}")

    # Final summary
    print("\n" + "="*70)
    print("SEASON 5 SUMMARY")
    print("="*70)
    print("\nExp 14: Synergy × Environment")
    for r in exp14_results:
        print(f"  {r['label']:>20s}  syn={r['n_syn']}D  fric={r['friction']:>5s}  "
              f"fit={r['fitness']:+8.2f}  r(Fx)={r['r_fx']:+.3f}")
    if len(exp14_results)>=2:
        syn_fit = exp14_results[1]["fitness"]
        env_fit = exp14_results[2]["fitness"] if len(exp14_results)>2 else 179
        if syn_fit > max(exp14_results[0]["fitness"], env_fit):
            print(f"\n  🔥 RESONANCE! Synergy+Env ({syn_fit:+.0f}) > both individual maxima!")
        else:
            print(f"\n  Interference: Synergy+Env ({syn_fit:+.0f}) does not exceed individual maxima")

    print(f"\nExp 15: Synergy Dimension Sweet Spot")
    for r in exp15_results:
        print(f"  {r['n_dims']:3d}D ({r['n_outputs']:2d} outputs)  fit={r['fitness']:+8.2f}  r(Fx)={r['r_fx']:+.3f}")
    # Find critical dimension
    if len(exp15_results) > 1:
        best_idx = max(range(len(exp15_results)), key=lambda i: exp15_results[i]["fitness"])
        print(f"\n  Sweet Spot: {exp15_results[best_idx]['n_dims']}D (fit={exp15_results[best_idx]['fitness']:+.0f})")
        # Find where fitness drops below 0 (DoF Trap)
        trap_dims = [r["n_dims"] for r in exp15_results if r["fitness"] < 0]
        if trap_dims:
            print(f"  DoF Trap activates at: {min(trap_dims)}D")
        else:
            print(f"  DoF Trap NOT triggered up to {max(dims)}D!")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
    print("\nAll Season 5 experiments complete!")


if __name__ == "__main__":
    main()
