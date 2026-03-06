"""
Extension Experiments: Generalizing the Symmetry Locks Theorem
==============================================================
Three experiments run sequentially:

Exp A - Generalization: Does the theorem hold for other asymmetries?
  A1: Stiffness asymmetry (k=30 vs k=3, equal mass)
  A2: Shape asymmetry (10x5x4 vs 5x5x8, equal mass)
  A3: Combined (mass 3:0.3 + stiffness 30:3)

Exp B - Long-term evolution: 500 generations for Extreme (10:1)
  Does fitness recover while maintaining differentiation?

Exp C - Self-evolving morphology: mass as genome
  Each body's mass is an evolvable gene. Does evolution invent asymmetry?

Autonomous: runs to completion, beeps when done.
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
FRICTION = 3.0; BASE_AMP = 30.0; DRAG = 0.4; SPRING_DAMP = 1.5
INPUT_SIZE = 7; HIDDEN_SIZE = 32; OUTPUT_SIZE = 3
N_W1 = INPUT_SIZE * HIDDEN_SIZE; N_B1 = HIDDEN_SIZE
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE; N_B2 = OUTPUT_SIZE
N_GENES_BASE = N_W1 + N_B1 + N_W2 + N_B2 + 1  # +1 for freq


def build_bodies(gx0, gy0, gz0, gx1, gy1, gz1, sp, gap):
    """Build two bodies with potentially different grid sizes."""
    nper0 = gx0*gy0*gz0; nper1 = gx1*gy1*gz1; nt = nper0 + nper1
    ap = np.zeros((nt, 3)); bi = np.zeros(nt, dtype=np.int64)
    # Body 0
    bw0 = (gx0-1)*sp; idx = 0
    for x in range(gx0):
        for y in range(gy0):
            for z in range(gz0):
                xp = -(gap/2+bw0) + x*sp
                ap[idx] = [xp, 2.0+y*sp, z*sp-(gz0-1)*sp/2]; bi[idx] = 0; idx += 1
    # Body 1
    bw1 = (gx1-1)*sp
    for x in range(gx1):
        for y in range(gy1):
            for z in range(gz1):
                xp = (gap/2+bw1) - x*sp
                ap[idx] = [xp, 2.0+y*sp, z*sp-(gz1-1)*sp/2]; bi[idx] = 1; idx += 1
    sa, sb, rl = [], [], []
    for b in range(2):
        m = np.where(bi==b)[0]; bp = ap[m]; tri = Delaunay(bp); edges = set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1,4): edges.add((min(m[s[i]],m[s[j]]),max(m[s[i]],m[s[j]])))
        for a,bb in edges: sa.append(a); sb.append(bb); rl.append(np.linalg.norm(ap[a]-ap[bb]))
    np_ = np.zeros_like(ap)
    for b in range(2):
        m = bi==b
        for d in range(3):
            vn,vx = ap[m,d].min(),ap[m,d].max()
            rng = vx - vn + 1e-8
            np_[m,d] = 2*(ap[m,d]-vn)/rng - 1
    return ap, np_, bi, np.array(sa), np.array(sb), np.array(rl), nper0, nper1, nt


@torch.no_grad()
def simulate(genomes, rp, npt, bit, sat, sbt, rlt, N, nper0, nsteps,
             per_body_mass=None, spring_k_per_body=None):
    B = genomes.shape[0]; pos = rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel = torch.zeros(B,N,3,device=DEVICE)
    idx=0; W1=genomes[:,idx:idx+N_W1].reshape(B,INPUT_SIZE,HIDDEN_SIZE);idx+=N_W1
    b1=genomes[:,idx:idx+N_B1].unsqueeze(1);idx+=N_B1
    W2=genomes[:,idx:idx+N_W2].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);idx+=N_W2
    b2=genomes[:,idx:idx+N_B2].unsqueeze(1);idx+=N_B2
    freq=genomes[:,idx].abs()
    sx=pos[:,:,0].mean(dim=1);bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False
    mass = torch.ones(N, device=DEVICE)
    if per_body_mass:
        mass[b0m] = per_body_mass[0]; mass[b1m] = per_body_mass[1]
    inv_mass = 1.0 / mass
    # Spring stiffness: default 30.0, or per-body
    sk = 30.0  # default
    # Build per-spring stiffness tensor if needed
    if spring_k_per_body:
        # Average stiffness of the two endpoints
        k_arr = torch.ones(N, device=DEVICE) * spring_k_per_body[0]
        k_arr[b1m] = spring_k_per_body[1]
        # Will compute per-spring below
        use_per_spring_k = True
    else:
        use_per_spring_k = False
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
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass.unsqueeze(0)
        pa=pos[:,csa];pb=pos[:,csb];d=pb-pa;di=torch.norm(d,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        # Spring stiffness
        if use_per_spring_k:
            # Mean stiffness of two endpoints
            per_k = (k_arr[csa] + k_arr[csb]) / 2.0
            per_k = per_k.unsqueeze(0).unsqueeze(2)  # (1, nsprings, 1)
            ft=per_k*s*dr+SPRING_DAMP*va*dr
        else:
            ft=sk*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=FRICTION*vel[:,:,0]*bl;f[:,:,2]-=FRICTION*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        acc = f * inv_mass.unsqueeze(0).unsqueeze(2)
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


def evolve(nsteps, gap, ngens, psz, per_body_mass=None, spring_k_per_body=None,
           body_grids=None, label="", n_genes=None):
    if body_grids is None:
        body_grids = ((10,5,4), (10,5,4))
    gx0,gy0,gz0 = body_grids[0]; gx1,gy1,gz1 = body_grids[1]
    sp = 0.35
    data = build_bodies(gx0,gy0,gz0, gx1,gy1,gz1, sp, gap)
    ap, np_, bi, sa, sb, rl, nper0, nper1, nt = data
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    if n_genes is None:
        n_genes = N_GENES_BASE
    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    pop=torch.randn(psz,n_genes,device=DEVICE)*0.3
    pop[:,:N_W1]*=s1/0.3;pop[:,N_W1:N_W1+N_B1]=0
    pop[:,N_W1+N_B1:N_W1+N_B1+N_W2]*=s2/0.3
    pop[:,N_W1+N_B1+N_W2:N_W1+N_B1+N_W2+N_B2]=0
    pop[:,N_GENES_BASE-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE); t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            # Extract mass from genome for Exp C
            if n_genes > N_GENES_BASE:
                mass_genes = pop[ix, N_GENES_BASE:]
                # Softmax to normalize total mass to 2.0
                mass_ratio = torch.softmax(mass_genes, dim=1)
                m0 = mass_ratio[:, 0] * 2.0; m1 = mass_ratio[:, 1] * 2.0
                # Use mean across batch for physics (batch-uniform mass)
                pbm = [m0.mean().item(), m1.mean().item()]
            else:
                pbm = per_body_mass
            f,_=simulate(pop[ix,:N_GENES_BASE],rp,npt,bit,sat,sbt,rlt,nt,nper0,nsteps,
                        pbm, spring_k_per_body)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]
        if gen%50==0: print(f"  [{label}] Gen {gen:3d}/{ngens}: best={pf[0].item():+.2f}")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,n_genes,device=DEVICE)*0.3
        fr[:,:N_W1]*=s1/0.3;fr[:,N_GENES_BASE-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np2.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE)
        p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE)
        p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,n_genes,device=DEVICE)<0.5
        ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,n_genes,device=DEVICE)<0.15
        ch+=torch.randn(nc,n_genes,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2
    elapsed=time.time()-t0
    print(f"  [{label}] Done: {elapsed/60:.1f}min | Best={pf[0].item():+.2f}")
    return pop[0].cpu().numpy(), pf[0].item(), data, elapsed


def replay_forces(genes, data, nsteps, per_body_mass=None, spring_k_per_body=None):
    """Replay best genome and track per-body forces. Returns r(Fx), r(Fy)."""
    ap, np_, bi, sa, sb, rl, nper0, nper1, nt = data
    N = nt
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    genome = torch.tensor(genes[:N_GENES_BASE], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    B=1; pos=rp.unsqueeze(0).clone(); vel=torch.zeros(B,N,3,device=DEVICE)
    gidx=0; W1=genome[:,gidx:gidx+N_W1].reshape(B,INPUT_SIZE,HIDDEN_SIZE);gidx+=N_W1
    b1g=genome[:,gidx:gidx+N_B1].unsqueeze(1);gidx+=N_B1
    W2=genome[:,gidx:gidx+N_W2].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);gidx+=N_W2
    b2g=genome[:,gidx:gidx+N_B2].unsqueeze(1);gidx+=N_B2
    freq=genome[:,gidx].abs()
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False
    mass = torch.ones(N, device=DEVICE)
    if per_body_mass:
        mass[b0m] = per_body_mass[0]; mass[b1m] = per_body_mass[1]
    # Spring stiffness
    sk = 30.0
    if spring_k_per_body:
        k_arr = torch.ones(N, device=DEVICE) * spring_k_per_body[0]
        k_arr[b1m] = spring_k_per_body[1]
        use_per_spring_k = True
    else:
        use_per_spring_k = False
    fx0_list, fx1_list, fy0_list, fy1_list = [], [], [], []
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
        freq_val = freq.item()
        sin_val = np.sin(2*np.pi*freq_val*t)
        cos_val = np.cos(2*np.pi*freq_val*t)
        st=torch.full((B,N,1), sin_val, device=DEVICE)
        ct=torch.full((B,N,1), cos_val, device=DEVICE)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        # Record per-body mean forces
        fx0_list.append(ext[0,b0m,0].mean().item())
        fx1_list.append(ext[0,b1m,0].mean().item())
        fy0_list.append(ext[0,b0m,1].mean().item())
        fy1_list.append(ext[0,b1m,1].mean().item())
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass.unsqueeze(0)
        pa=pos[:,csa];pb=pos[:,csb];d=pb-pa;di=torch.norm(d,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        if use_per_spring_k:
            per_k = (k_arr[csa] + k_arr[csb]) / 2.0
            per_k = per_k.unsqueeze(0).unsqueeze(2)
            ft=per_k*s*dr+SPRING_DAMP*va*dr
        else:
            ft=sk*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=FRICTION*vel[:,:,0]*bl;f[:,:,2]-=FRICTION*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        inv_mass = 1.0 / mass
        acc = f * inv_mass.unsqueeze(0).unsqueeze(2)
        vel+=acc*DT; vel.clamp_(-50, 50); pos+=vel*DT
    r_fx, _ = pearsonr(fx0_list, fx1_list)
    r_fy, _ = pearsonr(fy0_list, fy1_list)
    return r_fx, r_fy


def main():
    print("="*70)
    print("EXTENSION EXPERIMENTS: Generalizing the Symmetry Locks Theorem")
    print("="*70)
    t_start = time.time()
    NSTEPS = 600; GAP = 0.5; PSZ = 200
    results = {}

    # ================================================================
    # EXP A: GENERALIZATION
    # ================================================================
    print("\n" + "="*70)
    print("EXP A: GENERALIZATION (stiffness, shape, combined)")
    print("="*70)

    conditions_a = [
        {"label": "A0_mass_only",       "mass": [3.0, 0.3], "spring_k": None,       "grids": None,                     "desc": "Mass 10:1 (baseline)"},
        {"label": "A1_stiffness_only",  "mass": None,       "spring_k": [30.0, 3.0], "grids": None,                     "desc": "Stiffness 10:1"},
        {"label": "A2_shape_only",      "mass": None,       "spring_k": None,       "grids": ((10,5,4),(5,5,8)),        "desc": "Shape 200 vs 200 (different grid)"},
        {"label": "A3_mass_stiffness",  "mass": [3.0, 0.3], "spring_k": [30.0, 3.0], "grids": None,                     "desc": "Mass 10:1 + Stiffness 10:1"},
    ]

    res_a = []
    for cond in conditions_a:
        print(f"\n--- {cond['label']}: {cond['desc']} ---")
        genes, fitness, data, elapsed = evolve(
            NSTEPS, GAP, 150, PSZ,
            per_body_mass=cond["mass"],
            spring_k_per_body=cond["spring_k"],
            body_grids=cond["grids"],
            label=cond["label"]
        )
        r_fx, r_fy = replay_forces(genes, data, NSTEPS,
                                    per_body_mass=cond["mass"],
                                    spring_k_per_body=cond["spring_k"])
        print(f"  r(Fx)={r_fx:.3f}, r(Fy)={r_fy:.3f}, fitness={fitness:+.2f}")
        res_a.append({
            "label": cond["label"], "desc": cond["desc"],
            "fitness": fitness, "r_fx": round(r_fx, 3), "r_fy": round(r_fy, 3),
            "elapsed_min": round(elapsed/60, 1)
        })
    results["exp_a_generalization"] = res_a

    # ================================================================
    # EXP B: LONG-TERM EVOLUTION (500 gens)
    # ================================================================
    print("\n" + "="*70)
    print("EXP B: LONG-TERM EVOLUTION (500 gens, Extreme 10:1)")
    print("="*70)

    genes_b, fitness_b, data_b, elapsed_b = evolve(
        NSTEPS, GAP, 500, PSZ,
        per_body_mass=[3.0, 0.3],
        label="B_long500"
    )
    r_fx_b, r_fy_b = replay_forces(genes_b, data_b, NSTEPS, per_body_mass=[3.0, 0.3])
    print(f"  500-gen: r(Fx)={r_fx_b:.3f}, r(Fy)={r_fy_b:.3f}, fitness={fitness_b:+.2f}")
    results["exp_b_longterm"] = {
        "ngens": 500, "fitness": fitness_b,
        "r_fx": round(r_fx_b, 3), "r_fy": round(r_fy_b, 3),
        "elapsed_min": round(elapsed_b/60, 1),
        "baseline_150gen_fitness": 105.12,
        "improved": bool(fitness_b > 105.12)
    }

    # ================================================================
    # EXP C: SELF-EVOLVING MORPHOLOGY (mass in genome)
    # ================================================================
    print("\n" + "="*70)
    print("EXP C: SELF-EVOLVING MORPHOLOGY (mass as genome)")
    print("="*70)

    # 2 extra genes: mass_body0_logit, mass_body1_logit
    n_genes_c = N_GENES_BASE + 2
    genes_c, fitness_c, data_c, elapsed_c = evolve(
        NSTEPS, GAP, 150, PSZ,
        per_body_mass=None,  # Mass determined by genome
        label="C_morpho",
        n_genes=n_genes_c
    )
    # Extract evolved masses
    mass_logits = genes_c[N_GENES_BASE:]
    mass_softmax = np.exp(mass_logits) / np.exp(mass_logits).sum()
    evolved_mass_0 = mass_softmax[0] * 2.0
    evolved_mass_1 = mass_softmax[1] * 2.0
    evolved_ratio = max(evolved_mass_0, evolved_mass_1) / min(evolved_mass_0, evolved_mass_1)
    print(f"  Evolved masses: Body0={evolved_mass_0:.3f}, Body1={evolved_mass_1:.3f}")
    print(f"  Evolved ratio: {evolved_ratio:.2f}:1")

    # Replay with evolved masses
    r_fx_c, r_fy_c = replay_forces(genes_c, data_c, NSTEPS,
                                    per_body_mass=[float(evolved_mass_0), float(evolved_mass_1)])
    print(f"  r(Fx)={r_fx_c:.3f}, r(Fy)={r_fy_c:.3f}, fitness={fitness_c:+.2f}")
    results["exp_c_morpho"] = {
        "evolved_mass_0": round(float(evolved_mass_0), 4),
        "evolved_mass_1": round(float(evolved_mass_1), 4),
        "evolved_ratio": round(float(evolved_ratio), 2),
        "fitness": fitness_c,
        "r_fx": round(r_fx_c, 3), "r_fy": round(r_fy_c, 3),
        "symmetry_broken": bool(evolved_ratio > 1.5),
        "elapsed_min": round(elapsed_c/60, 1)
    }

    # ================================================================
    # SUMMARY FIGURE
    # ================================================================
    print("\n" + "="*70)
    print("Generating summary figure...")
    print("="*70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Extension Experiments: Generalizing the Symmetry Locks Theorem",
                 fontsize=14, fontweight='bold')

    # Panel 1: Exp A - Generalization bar chart
    ax = axes[0]
    labels_a = [r["label"].split("_",1)[1] for r in res_a]
    rfx_a = [r["r_fx"] for r in res_a]
    fit_a = [r["fitness"] for r in res_a]
    colors_a = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    x_pos = np.arange(len(labels_a))
    bars = ax.bar(x_pos, rfx_a, color=colors_a, alpha=0.8, width=0.6)
    ax.axhline(y=0.742, color='gray', linestyle='--', alpha=0.5, label='Symmetric baseline')
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Differentiation threshold')
    ax.set_xticks(x_pos); ax.set_xticklabels(labels_a, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("r(Fx)"); ax.set_title("A: Generalization\n(Which asymmetries break synchronization?)")
    ax.legend(fontsize=8); ax.set_ylim(-0.3, 1.0)
    for bar, val, f in zip(bars, rfx_a, fit_a):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                f"r={val:.3f}\nfit={f:+.0f}", ha='center', va='bottom', fontsize=8)

    # Panel 2: Exp B - Long-term evolution
    ax = axes[1]
    gen_labels = ['150 gen\n(original)', '500 gen\n(extended)']
    fitness_vals = [105.12, fitness_b]
    rfx_vals = [0.020, r_fx_b]
    x_b = np.arange(2)
    ax2 = ax.twinx()
    ax.bar(x_b[0]-0.15, fitness_vals[0], 0.3, color='#e74c3c', alpha=0.5, label='Fitness (150g)')
    ax.bar(x_b[1]-0.15, fitness_vals[1], 0.3, color='#e74c3c', alpha=0.9, label='Fitness (500g)')
    ax2.bar(x_b[0]+0.15, rfx_vals[0], 0.3, color='#3498db', alpha=0.5, label='r(Fx) (150g)')
    ax2.bar(x_b[1]+0.15, rfx_vals[1], 0.3, color='#3498db', alpha=0.9, label='r(Fx) (500g)')
    ax.set_xticks(x_b); ax.set_xticklabels(gen_labels)
    ax.set_ylabel("Fitness", color='#e74c3c'); ax2.set_ylabel("r(Fx)", color='#3498db')
    ax.set_title(f"B: Long-term Evolution\n(500 gen: fit={fitness_b:+.1f}, r={r_fx_b:.3f})")
    ax.set_ylim(0, max(fitness_vals)*1.3); ax2.set_ylim(-0.1, 1.0)

    # Panel 3: Exp C - Self-evolving morphology
    ax = axes[2]
    ax.bar([0, 1], [evolved_mass_0, evolved_mass_1], color=['#e74c3c', '#3498db'],
           alpha=0.8, width=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Symmetric (1.0)')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Body 0\n(evolved)', 'Body 1\n(evolved)'])
    ax.set_ylabel("Mass")
    ax.set_title(f"C: Self-Evolving Morphology\nRatio={evolved_ratio:.2f}:1, r(Fx)={r_fx_c:.3f}")
    ax.legend(fontsize=8)
    for i, v in enumerate([evolved_mass_0, evolved_mass_1]):
        ax.text(i, v+0.02, f"{v:.3f}", ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "extension_experiments.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # Save JSON log
    total_time = (time.time() - t_start) / 60
    results["total_time_min"] = round(total_time, 1)
    log_path = os.path.join(RESULTS_DIR, "extension_experiments_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Log saved: {log_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nExp A: Generalization")
    for r in res_a:
        status = "DIFFERENTIATED" if r["r_fx"] < 0.3 else "Synchronized"
        print(f"  {r['desc']:40s} r(Fx)={r['r_fx']:.3f}  [{status}]")
    print(f"\nExp B: Long-term (500 gen)")
    improved = "YES" if fitness_b > 105.12 else "NO"
    print(f"  Fitness: {fitness_b:+.2f} (vs 105.12 @ 150gen)  Improved: {improved}")
    print(f"  r(Fx): {r_fx_b:.3f}  (differentiation maintained: {r_fx_b < 0.3})")
    print(f"\nExp C: Self-evolving morphology")
    print(f"  Evolved masses: {evolved_mass_0:.3f} / {evolved_mass_1:.3f}  (ratio {evolved_ratio:.2f}:1)")
    print(f"  Symmetry broken: {evolved_ratio > 1.5}")
    print(f"\nTotal time: {total_time:.1f} min")

    # Beep
    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
    print("\nDone!")


if __name__ == "__main__":
    main()
