"""
Season 4B: Muscle Synergy (DoF Trap Breakthrough) + Environmental Reversal
===========================================================================
Exp 12: 1D Mass Shift (Muscle Synergy) - compress mass control to 1 dimension
Exp 13: Environmental Reversal (Plasticity Test) - flip friction mid-simulation
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
INPUT_SIZE = 7; HIDDEN_SIZE = 32
N_W1 = INPUT_SIZE * HIDDEN_SIZE


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


def get_n_genes(output_size):
    return INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*output_size + output_size + 1


@torch.no_grad()
def simulate_synergy(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps,
                     mode="fixed", base_mass=1.0, synergy_strength=0.5,
                     friction_map=None, flip_friction_at=None):
    """
    mode:
      "fixed"   - standard 3-output, no mass control
      "synergy" - 4 outputs, 4th = 1D global mass shift (body0 heavy ↔ body1 heavy)
      "perdim"  - 4 outputs, 4th = per-particle mass (original DoF Trap)
    """
    OUTPUT_SIZE = 4 if mode in ("synergy", "perdim") else 3
    N_GENES = get_n_genes(OUTPUT_SIZE)
    FRICTION_DEFAULT = 3.0

    B = genomes.shape[0]; pos = rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel = torch.zeros(B,N,3,device=DEVICE)
    idx=0; W1=genomes[:,idx:idx+N_W1].reshape(B,INPUT_SIZE,HIDDEN_SIZE);idx+=N_W1
    b1=genomes[:,idx:idx+HIDDEN_SIZE].unsqueeze(1);idx+=HIDDEN_SIZE
    W2=genomes[:,idx:idx+HIDDEN_SIZE*OUTPUT_SIZE].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);idx+=HIDDEN_SIZE*OUTPUT_SIZE
    b2=genomes[:,idx:idx+OUTPUT_SIZE].unsqueeze(1);idx+=OUTPUT_SIZE
    freq=genomes[:,idx].abs()
    sx=pos[:,:,0].mean(dim=1);bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False

    # Base mass
    mass = torch.ones(B,N,device=DEVICE) * base_mass

    # Friction
    fric = torch.full((N,), FRICTION_DEFAULT, device=DEVICE)
    fric_flipped = None
    if friction_map:
        fric[b0m] = friction_map['body0_friction']
        fric[b1m] = friction_map['body1_friction']
        if flip_friction_at is not None:
            fric_flipped = torch.full((N,), FRICTION_DEFAULT, device=DEVICE)
            fric_flipped[b0m] = friction_map['body1_friction']  # swapped!
            fric_flipped[b1m] = friction_map['body0_friction']

    for step in range(nsteps):
        t = step * DT

        # Handle friction flip
        current_fric = fric
        if flip_friction_at is not None and fric_flipped is not None and step >= flip_friction_at:
            current_fric = fric_flipped

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

        # Mass control
        if mode == "synergy":
            # 4th output averaged across particles → single 1D signal per individual
            shift = o[:,:,3].mean(dim=1)  # (B,) scalar, range [-1, 1]
            # Distribute: body0 gets heavier when shift>0, body1 when shift<0
            mass_new = torch.ones(B,N,device=DEVICE) * base_mass
            mass_new[:, b0m] = base_mass * (1.0 + synergy_strength * shift.unsqueeze(1))
            mass_new[:, b1m] = base_mass * (1.0 - synergy_strength * shift.unsqueeze(1))
            mass_new = mass_new.clamp(min=0.1)
            mass = mass_new
        elif mode == "perdim":
            # Per-particle mass via softmax (original DoF Trap approach)
            mass_raw = o[:,:,3]  # (B, N)
            mass_soft = torch.softmax(mass_raw, dim=1) * N * base_mass
            mass = mass_soft.clamp(min=0.01)

        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d=pb-pa;di=torch.norm(d,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=current_fric.unsqueeze(0)*vel[:,:,0]*bl
        f[:,:,2]-=current_fric.unsqueeze(0)*vel[:,:,2]*bl
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


def replay_rfx_synergy(genes, data, nsteps, mode, base_mass=1.0,
                       synergy_strength=0.5, friction_map=None, flip_friction_at=None):
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt
    OUTPUT_SIZE = 4 if mode in ("synergy", "perdim") else 3
    N_GENES_L = get_n_genes(OUTPUT_SIZE)
    FRICTION_DEFAULT = 3.0
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    genome=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    gidx=0;W1=genome[:,gidx:gidx+N_W1].reshape(B,INPUT_SIZE,HIDDEN_SIZE);gidx+=N_W1
    b1g=genome[:,gidx:gidx+HIDDEN_SIZE].unsqueeze(1);gidx+=HIDDEN_SIZE
    W2=genome[:,gidx:gidx+HIDDEN_SIZE*OUTPUT_SIZE].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);gidx+=HIDDEN_SIZE*OUTPUT_SIZE
    b2g=genome[:,gidx:gidx+OUTPUT_SIZE].unsqueeze(1);gidx+=OUTPUT_SIZE
    freq_val=genome[:,gidx].abs().item()
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False
    mass=torch.ones(B,N,device=DEVICE)*base_mass
    fric=torch.full((N,),FRICTION_DEFAULT,device=DEVICE)
    fric_flipped=None
    if friction_map:
        fric[b0m]=friction_map['body0_friction'];fric[b1m]=friction_map['body1_friction']
        if flip_friction_at is not None:
            fric_flipped=torch.full((N,),FRICTION_DEFAULT,device=DEVICE)
            fric_flipped[b0m]=friction_map['body1_friction']
            fric_flipped[b1m]=friction_map['body0_friction']
    fx0_list,fx1_list=[],[]
    for step in range(nsteps):
        t=step*DT
        current_fric=fric
        if flip_friction_at is not None and fric_flipped is not None and step>=flip_friction_at:
            current_fric=fric_flipped
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
        if mode=="synergy":
            shift=o[:,:,3].mean(dim=1)
            mass_new=torch.ones(B,N,device=DEVICE)*base_mass
            mass_new[:,b0m]=base_mass*(1.0+synergy_strength*shift.unsqueeze(1))
            mass_new[:,b1m]=base_mass*(1.0-synergy_strength*shift.unsqueeze(1))
            mass=mass_new.clamp(min=0.1)
        elif mode=="perdim":
            mass_raw=o[:,:,3];mass_soft=torch.softmax(mass_raw,dim=1)*N*base_mass
            mass=mass_soft.clamp(min=0.01)
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        fx0_list.append(ext[0,b0m,0].mean().item());fx1_list.append(ext[0,b1m,0].mean().item())
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d=pb-pa;di=torch.norm(d,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=current_fric.unsqueeze(0)*vel[:,:,0]*bl;f[:,:,2]-=current_fric.unsqueeze(0)*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        inv_mass=1.0/mass.clamp(min=0.01);acc=f*inv_mass.unsqueeze(2)
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
    if np.std(fx0_list)<1e-10 or np.std(fx1_list)<1e-10: return 0.0
    r_fx,_=pearsonr(fx0_list,fx1_list)
    return r_fx


def evolve_synergy(data, nsteps, ngens, psz, mode, label,
                   base_mass=1.0, synergy_strength=0.5,
                   friction_map=None, flip_friction_at=None):
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt
    OUTPUT_SIZE = 4 if mode in ("synergy", "perdim") else 3
    N_GENES_L = get_n_genes(OUTPUT_SIZE)
    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    pop=torch.randn(psz,N_GENES_L,device=DEVICE)*0.3
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
            f,_=simulate_synergy(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,
                                 mode,base_mass,synergy_strength,friction_map,flip_friction_at)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]
        if gen%50==0 or gen==ngens-1:
            gen_log.append(gen);fitness_log.append(pf[0].item())
            elapsed=time.time()-t0
            print(f"  [{label}] Gen {gen:4d}/{ngens}: fit={pf[0].item():+.2f}  ({elapsed/60:.1f}min)")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,N_GENES_L,device=DEVICE)*0.3
        fr[:,:N_W1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE)
        p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE)
        p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,N_GENES_L,device=DEVICE)<0.5
        ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,N_GENES_L,device=DEVICE)<0.15
        ch+=torch.randn(nc,N_GENES_L,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2
    total=(time.time()-t0)/60;best_genes=pop[0].cpu().numpy()
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return best_genes, gen_log, fitness_log, total


def main():
    NSTEPS=600;GAP=0.5;PSZ=200
    gx,gy,gz,sp=10,5,4,0.35
    data=build_bodies(gx,gy,gz,sp,GAP)
    results={}

    # ================================================================
    print("="*70)
    print("EXP 12: MUSCLE SYNERGY - DoF Trap Breakthrough?")
    print("="*70)

    exp12_results = []

    # Control: Fixed 3-output, 300 gen
    NGENS_12 = 300
    print("\n--- 12a: Fixed 3-output (control, 300gen) ---")
    best_g, gl, fl, elapsed = evolve_synergy(data, NSTEPS, NGENS_12, PSZ, "fixed", "12a_fixed")
    r_fx = replay_rfx_synergy(best_g, data, NSTEPS, "fixed")
    exp12_results.append({"label": "fixed_3out", "fitness": round(fl[-1],2), "r_fx": round(r_fx,3), "elapsed": round(elapsed,1)})
    print(f"  r(Fx)={r_fx:.3f}, fit={fl[-1]:+.2f}")

    # Synergy: 4-output with 1D mass shift, 300 gen
    print("\n--- 12b: Muscle Synergy 4-output (1D shift, 300gen) ---")
    best_g, gl, fl, elapsed = evolve_synergy(data, NSTEPS, NGENS_12, PSZ, "synergy", "12b_synergy",
                                             synergy_strength=0.5)
    r_fx = replay_rfx_synergy(best_g, data, NSTEPS, "synergy", synergy_strength=0.5)
    exp12_results.append({"label": "synergy_1D_s0.5", "fitness": round(fl[-1],2), "r_fx": round(r_fx,3), "elapsed": round(elapsed,1)})
    print(f"  r(Fx)={r_fx:.3f}, fit={fl[-1]:+.2f}")

    # Synergy with stronger shift
    print("\n--- 12c: Muscle Synergy 4-output (1D shift strong, 300gen) ---")
    best_g, gl, fl, elapsed = evolve_synergy(data, NSTEPS, NGENS_12, PSZ, "synergy", "12c_synergy_strong",
                                             synergy_strength=0.9)
    r_fx = replay_rfx_synergy(best_g, data, NSTEPS, "synergy", synergy_strength=0.9)
    exp12_results.append({"label": "synergy_1D_s0.9", "fitness": round(fl[-1],2), "r_fx": round(r_fx,3), "elapsed": round(elapsed,1)})
    print(f"  r(Fx)={r_fx:.3f}, fit={fl[-1]:+.2f}")

    # Per-dim (original DoF Trap, for comparison), 300 gen
    print("\n--- 12d: Per-particle mass (DoF Trap reproduced, 300gen) ---")
    best_g, gl, fl, elapsed = evolve_synergy(data, NSTEPS, NGENS_12, PSZ, "perdim", "12d_perdim")
    r_fx = replay_rfx_synergy(best_g, data, NSTEPS, "perdim")
    exp12_results.append({"label": "perdim_DoF_trap", "fitness": round(fl[-1],2), "r_fx": round(r_fx,3), "elapsed": round(elapsed,1)})
    print(f"  r(Fx)={r_fx:.3f}, fit={fl[-1]:+.2f}")

    results["exp12_muscle_synergy"] = exp12_results

    # ================================================================
    print("\n" + "="*70)
    print("EXP 13: ENVIRONMENTAL REVERSAL - Plasticity Test")
    print("="*70)

    NGENS_13 = 150
    fmap_asym = {"body0_friction": 0.1, "body1_friction": 5.0}
    exp13_results = []

    # 13a: Normal asymmetric friction (control)
    print("\n--- 13a: Asymmetric friction, no flip (control) ---")
    best_normal, gl, fl, elapsed = evolve_synergy(data, NSTEPS, NGENS_13, PSZ, "fixed", "13a_normal",
                                                  friction_map=fmap_asym)
    r_normal = replay_rfx_synergy(best_normal, data, NSTEPS, "fixed", friction_map=fmap_asym)
    exp13_results.append({"label": "normal_asym", "fitness": round(fl[-1],2), "r_fx": round(r_normal,3),
                          "elapsed": round(elapsed,1), "flip": "none"})
    print(f"  r(Fx)={r_normal:.3f}, fit={fl[-1]:+.2f}")

    # 13b: Replay the normal-trained genome with FLIPPED friction
    print("\n--- 13b: Replay normal genome with FLIPPED friction ---")
    # Full flip: body0 gets 5.0, body1 gets 0.1 (swapped from training)
    fmap_flipped = {"body0_friction": 5.0, "body1_friction": 0.1}
    # Replay using the normal genome but fully flipped friction
    r_flipped = replay_rfx_synergy(best_normal, data, NSTEPS, "fixed", friction_map=fmap_flipped)
    # Also compute fitness for the flipped replay
    rp_t = torch.tensor(data[0], dtype=torch.float32, device=DEVICE)
    npt_t = torch.tensor(data[1], dtype=torch.float32, device=DEVICE)
    bit_t = torch.tensor(data[2], dtype=torch.long, device=DEVICE)
    sat_t = torch.tensor(data[3], dtype=torch.long, device=DEVICE)
    sbt_t = torch.tensor(data[4], dtype=torch.long, device=DEVICE)
    rlt_t = torch.tensor(data[5], dtype=torch.float32, device=DEVICE)
    gen_t = torch.tensor(best_normal, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    fit_flipped, _ = simulate_synergy(gen_t, rp_t, npt_t, bit_t, sat_t, sbt_t, rlt_t,
                                      data[7], data[6], NSTEPS, "fixed",
                                      friction_map=fmap_flipped)
    exp13_results.append({"label": "flipped_replay", "fitness": round(fit_flipped[0].item(),2),
                          "r_fx": round(r_flipped,3), "elapsed": 0, "flip": "full_flip"})
    print(f"  r(Fx)={r_flipped:.3f}, fit={fit_flipped[0].item():+.2f}")

    # 13c: Replay with mid-simulation flip (step 300 of 600)
    print("\n--- 13c: Replay with mid-sim friction flip (step 300) ---")
    r_midflip = replay_rfx_synergy(best_normal, data, NSTEPS, "fixed",
                                   friction_map=fmap_asym, flip_friction_at=300)
    fit_midflip, _ = simulate_synergy(gen_t, rp_t, npt_t, bit_t, sat_t, sbt_t, rlt_t,
                                      data[7], data[6], NSTEPS, "fixed",
                                      friction_map=fmap_asym, flip_friction_at=300)
    exp13_results.append({"label": "midflip_replay", "fitness": round(fit_midflip[0].item(),2),
                          "r_fx": round(r_midflip,3), "elapsed": 0, "flip": "mid_300"})
    print(f"  r(Fx)={r_midflip:.3f}, fit={fit_midflip[0].item():+.2f}")

    # 13d: TRAINED with mid-sim flip (evolve with flip during training)
    print("\n--- 13d: Evolved WITH mid-sim flip (adapts to both environments) ---")
    best_plastic, gl, fl, elapsed = evolve_synergy(data, NSTEPS, NGENS_13, PSZ, "fixed", "13d_plastic",
                                                   friction_map=fmap_asym, flip_friction_at=300)
    r_plastic = replay_rfx_synergy(best_plastic, data, NSTEPS, "fixed",
                                   friction_map=fmap_asym, flip_friction_at=300)
    exp13_results.append({"label": "plastic_trained", "fitness": round(fl[-1],2),
                          "r_fx": round(r_plastic,3), "elapsed": round(elapsed,1), "flip": "trained_with_flip"})
    print(f"  r(Fx)={r_plastic:.3f}, fit={fl[-1]:+.2f}")

    # 13e: symmetric friction control (no asymmetry)
    print("\n--- 13e: Symmetric friction (baseline control) ---")
    best_sym, gl, fl, elapsed = evolve_synergy(data, NSTEPS, NGENS_13, PSZ, "fixed", "13e_sym_control")
    r_sym = replay_rfx_synergy(best_sym, data, NSTEPS, "fixed")
    exp13_results.append({"label": "symmetric_control", "fitness": round(fl[-1],2),
                          "r_fx": round(r_sym,3), "elapsed": round(elapsed,1), "flip": "none"})
    print(f"  r(Fx)={r_sym:.3f}, fit={fl[-1]:+.2f}")

    results["exp13_environmental_reversal"] = exp13_results

    # ================================================================
    # SUMMARY FIGURE
    # ================================================================
    print("\nGenerating summary figure...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Season 4B: Muscle Synergy & Environmental Reversal",
                 fontsize=14, fontweight="bold")

    # Panel 1: Muscle Synergy
    ax = axes[0]
    labels_12 = [r["label"].replace("_", "\n") for r in exp12_results]
    fits_12 = [r["fitness"] for r in exp12_results]
    colors_12 = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    bars = ax.bar(range(len(exp12_results)), fits_12, color=colors_12, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(exp12_results)))
    ax.set_xticklabels(labels_12, fontsize=7)
    ax.set_ylabel("Fitness")
    ax.set_title("Exp 12: Muscle Synergy (DoF Trap Breakthrough?)")
    for i, r in enumerate(exp12_results):
        y_pos = max(fits_12[i] + 3, 5)
        ax.text(i, y_pos, f"fit={fits_12[i]:+.0f}\nr={r['r_fx']:.3f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    # Add horizontal line for fixed baseline
    if len(exp12_results) > 0:
        ax.axhline(y=exp12_results[0]["fitness"], color="green", linestyle="--", alpha=0.4,
                   label=f"Fixed baseline ({exp12_results[0]['fitness']:+.0f})")
    ax.legend(fontsize=8)

    # Panel 2: Environmental Reversal
    ax = axes[1]
    labels_13 = [r["label"].replace("_", "\n") for r in exp13_results]
    fits_13 = [r["fitness"] for r in exp13_results]
    rfx_13 = [r["r_fx"] for r in exp13_results]
    colors_13 = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db", "#95a5a6"]
    x_pos = np.arange(len(exp13_results))
    bars = ax.bar(x_pos - 0.15, rfx_13, 0.3, color=colors_13[:len(exp13_results)], alpha=0.8, label="r(Fx)")
    ax2 = ax.twinx()
    ax2.bar(x_pos + 0.15, fits_13, 0.3, color=colors_13[:len(exp13_results)], alpha=0.35, label="Fitness")
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_13, fontsize=6)
    ax.set_ylabel("r(Fx)")
    ax2.set_ylabel("Fitness", alpha=0.5)
    ax.set_ylim(-0.5, 1.0)
    for i, r in enumerate(exp13_results):
        ax.text(i, rfx_13[i]+0.05, f"r={rfx_13[i]:.2f}", ha="center", fontsize=7)
        ax.text(i, rfx_13[i]-0.12, f"fit={fits_13[i]:+.0f}", ha="center", fontsize=6, color="gray")
    ax.set_title("Exp 13: Environmental Reversal (Plasticity)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "season4b_experiments.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")

    log_path = os.path.join(RESULTS_DIR, "season4b_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Log saved: {log_path}")

    # Final summary
    print("\n" + "="*70)
    print("SEASON 4B SUMMARY")
    print("="*70)
    print("\nExp 12: Muscle Synergy (DoF Trap)")
    for r in exp12_results:
        print(f"  {r['label']:>25s}  fit={r['fitness']:+8.2f}  r(Fx)={r['r_fx']:+.3f}")
    synergy_success = any(r["fitness"] > 0 and r["label"].startswith("synergy") for r in exp12_results)
    print(f"\n  DoF Trap Breakthrough: {'YES! Synergy works!' if synergy_success else 'No (Trap holds)'}")

    print(f"\nExp 13: Environmental Reversal")
    for r in exp13_results:
        print(f"  {r['label']:>20s}  fit={r['fitness']:+8.2f}  r(Fx)={r['r_fx']:+.3f}  flip={r['flip']}")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
    print("\nAll Season 4B experiments complete!")


if __name__ == "__main__":
    main()
