"""
Season 2 Experiments: Morphological Annealing, Memory, and 3-Body Extension
============================================================================
Exp D: Morphological Annealing - Anneal mass ratio from 10:1 to 1:1 over 500 gens
Exp E: Differentiation Memory - Evolve under 10:1, then switch to 1:1
Exp F: 3-Body Extension - Test Symmetry Locks with 3 bodies

Baseline: symmetric 1:1 for 500 gens (control)
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
FRICTION = 3.0; BASE_AMP = 30.0; DRAG = 0.4; SPRING_K = 30.0; SPRING_DAMP = 1.5
INPUT_SIZE = 7; HIDDEN_SIZE = 32; OUTPUT_SIZE = 3
N_W1 = INPUT_SIZE * HIDDEN_SIZE; N_B1 = HIDDEN_SIZE
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE; N_B2 = OUTPUT_SIZE
N_GENES = N_W1 + N_B1 + N_W2 + N_B2 + 1


# ========================================================================
# Core functions (2-body)
# ========================================================================
def build_bodies_2(gx, gy, gz, sp, gap):
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


def build_bodies_3(gx, gy, gz, sp, gap):
    """Build 3 bodies arranged in a row with spacing gap."""
    nper = gx*gy*gz; nt = nper*3; ap = np.zeros((nt,3)); bi = np.zeros(nt, dtype=np.int64)
    bw = (gx-1)*sp; idx = 0
    offsets = [-(gap+bw), 0.0, gap+bw]
    for b in range(3):
        for x in range(gx):
            for y in range(gy):
                for z in range(gz):
                    xp = offsets[b] + x*sp - bw/2
                    ap[idx] = [xp, 2.0+y*sp, z*sp-(gz-1)*sp/2]; bi[idx] = b; idx += 1
    sa, sb, rl = [], [], []
    for b in range(3):
        m = np.where(bi==b)[0]; bp = ap[m]; tri = Delaunay(bp); edges = set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1,4): edges.add((min(m[s[i]],m[s[j]]),max(m[s[i]],m[s[j]])))
        for a,bb in edges: sa.append(a);sb.append(bb);rl.append(np.linalg.norm(ap[a]-ap[bb]))
    np_ = np.zeros_like(ap)
    for b in range(3):
        m = bi==b
        for d in range(3):
            vn,vx = ap[m,d].min(),ap[m,d].max(); np_[m,d] = 2*(ap[m,d]-vn)/(vx-vn+1e-8)-1
    return ap, np_, bi, np.array(sa), np.array(sb), np.array(rl), nper, nt


@torch.no_grad()
def simulate(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps,
             per_body_mass=None, n_bodies=2):
    B = genomes.shape[0]; pos = rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel = torch.zeros(B,N,3,device=DEVICE)
    idx=0; W1=genomes[:,idx:idx+N_W1].reshape(B,INPUT_SIZE,HIDDEN_SIZE);idx+=N_W1
    b1=genomes[:,idx:idx+N_B1].unsqueeze(1);idx+=N_B1
    W2=genomes[:,idx:idx+N_W2].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);idx+=N_W2
    b2=genomes[:,idx:idx+N_B2].unsqueeze(1);idx+=N_B2
    freq=genomes[:,idx].abs()
    sx=pos[:,:,0].mean(dim=1)
    # body_id: for 3 bodies, use 0.0, 0.5, 1.0
    if n_bodies == 2:
        bid = bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    else:
        bid = (bit.float() / (n_bodies - 1)).unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE)
    cd=False
    mass = torch.ones(N, device=DEVICE)
    if per_body_mass:
        for b_idx in range(n_bodies):
            bm = bit == b_idx
            mass[bm] = per_body_mass[b_idx]
    inv_mass = 1.0 / mass
    # body masks for cohesion (pairs)
    body_masks = [(bit==b_idx) for b_idx in range(n_bodies)]
    for step in range(nsteps):
        t=step*DT
        if not cd and step%10==0:
            # Connect nearby bodies
            for b1_idx in range(n_bodies):
                for b2_idx in range(b1_idx+1, n_bodies):
                    bi1 = body_masks[b1_idx].nonzero(as_tuple=True)[0]
                    bi2 = body_masks[b2_idx].nonzero(as_tuple=True)[0]
                    p1_=pos[0,bi1]; p2_=pos[0,bi2]
                    ds=torch.cdist(p1_,p2_)
                    cl=(ds<1.2).nonzero(as_tuple=False)
                    if cl.shape[0]>0:
                        nn_=min(cl.shape[0],300)
                        csa=torch.cat([csa,bi1[cl[:nn_,0]]])
                        csb=torch.cat([csb,bi2[cl[:nn_,1]]])
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
        ft=SPRING_K*s*dr+SPRING_DAMP*va*dr
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
    spp=((sp_-8.0*(n_bodies/2)).clamp(min=0)*1.5).sum(dim=1)
    bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
    # Cohesion for all body pairs
    coh = torch.zeros(B, device=DEVICE)
    for b1_idx in range(n_bodies):
        for b2_idx in range(b1_idx+1, n_bodies):
            c1 = pos[:,body_masks[b1_idx]].mean(dim=1)
            c2 = pos[:,body_masks[b2_idx]].mean(dim=1)
            coh += torch.clamp(3.0-torch.norm(c1-c2,dim=1),min=0)*2.0
    fitness = disp-dz-spp-bw-ep+coh
    fitness = torch.where(torch.isnan(fitness), torch.tensor(-9999.0, device=DEVICE), fitness)
    return fitness, disp


def replay_rfx_pair(genes, data, nsteps, per_body_mass, body_a=0, body_b=1, n_bodies=2):
    """Replay to get r(Fx) between two specific bodies."""
    ap, np_, bi, sa, sb, rl, nper, nt = data
    N = nt
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    genome = torch.tensor(genes, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    B=1; pos=rp.unsqueeze(0).clone(); vel=torch.zeros(B,N,3,device=DEVICE)
    gidx=0; W1=genome[:,gidx:gidx+N_W1].reshape(B,INPUT_SIZE,HIDDEN_SIZE);gidx+=N_W1
    b1g=genome[:,gidx:gidx+N_B1].unsqueeze(1);gidx+=N_B1
    W2=genome[:,gidx:gidx+N_W2].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);gidx+=N_W2
    b2g=genome[:,gidx:gidx+N_B2].unsqueeze(1);gidx+=N_B2
    freq_val=genome[:,gidx].abs().item()
    if n_bodies == 2:
        bid = bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    else:
        bid = (bit.float()/(n_bodies-1)).unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE)
    cd=False
    mass = torch.ones(N, device=DEVICE)
    body_masks = [(bit==b) for b in range(n_bodies)]
    if per_body_mass:
        for b in range(n_bodies):
            mass[body_masks[b]] = per_body_mass[b]
    bam = body_masks[body_a]; bbm = body_masks[body_b]
    fx_a, fx_b = [], []
    for step in range(nsteps):
        t=step*DT
        if not cd and step%10==0:
            for b1i in range(n_bodies):
                for b2i in range(b1i+1, n_bodies):
                    bi1=body_masks[b1i].nonzero(as_tuple=True)[0]
                    bi2=body_masks[b2i].nonzero(as_tuple=True)[0]
                    ds=torch.cdist(pos[0,bi1],pos[0,bi2])
                    cl=(ds<1.2).nonzero(as_tuple=False)
                    if cl.shape[0]>0:
                        nn_=min(cl.shape[0],300)
                        csa=torch.cat([csa,bi1[cl[:nn_,0]]])
                        csb=torch.cat([csb,bi2[cl[:nn_,1]]])
                        crl=torch.cat([crl,ds[cl[:nn_,0],cl[:nn_,1]]])
                        comb=torch.ones(B,1,1,device=DEVICE);cd=True
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
        fx_a.append(ext[0,bam,0].mean().item())
        fx_b.append(ext[0,bbm,0].mean().item())
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass.unsqueeze(0)
        pa=pos[:,csa];pb=pos[:,csb];d=pb-pa;di=torch.norm(d,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=FRICTION*vel[:,:,0]*bl;f[:,:,2]-=FRICTION*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        inv_mass = 1.0 / mass
        acc = f * inv_mass.unsqueeze(0).unsqueeze(2)
        vel+=acc*DT; vel.clamp_(-50, 50); pos+=vel*DT
    if np.std(fx_a) < 1e-10 or np.std(fx_b) < 1e-10:
        return 0.0
    r_fx, _ = pearsonr(fx_a, fx_b)
    return r_fx


def evolve(data, nsteps, ngens, psz, per_body_mass_fn, probe_interval=50,
           label="", n_bodies=2, init_pop=None, init_pf=None):
    """Generic evolution loop with mass schedule and probing.
    per_body_mass_fn(gen) -> list of masses for each body.
    """
    ap,np_,bi,sa,sb,rl,nper,nt = data
    N = nt
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)

    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    if init_pop is not None:
        pop = init_pop.clone(); pf = init_pf.clone()
        psz = pop.shape[0]
    else:
        pop=torch.randn(psz,N_GENES,device=DEVICE)*0.3
        pop[:,:N_W1]*=s1/0.3;pop[:,N_W1:N_W1+N_B1]=0
        pop[:,N_W1+N_B1:N_W1+N_B1+N_W2]*=s2/0.3
        pop[:,N_W1+N_B1+N_W2:N_W1+N_B1+N_W2+N_B2]=0
        pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
        pf=torch.full((psz,),float('-inf'),device=DEVICE)

    gen_log, fitness_log, rfx_log, mass_log = [], [], [], []
    t0 = time.time()

    for gen in range(ngens):
        masses = per_body_mass_fn(gen)

        # Re-evaluate ALL when mass changes
        if gen == 0 or per_body_mass_fn(gen) != per_body_mass_fn(gen-1):
            pf = torch.full((psz,), float('-inf'), device=DEVICE)

        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,_=simulate(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,masses,n_bodies)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]

        if gen % probe_interval == 0 or gen == ngens-1:
            best_genes = pop[0].cpu().numpy()
            r_fx = replay_rfx_pair(best_genes, data, nsteps, masses,
                                    body_a=0, body_b=min(1, n_bodies-1), n_bodies=n_bodies)
            gen_log.append(gen); fitness_log.append(pf[0].item())
            rfx_log.append(r_fx)
            mass_log.append(masses[0] / masses[min(1, n_bodies-1)] if masses[min(1, n_bodies-1)] > 0 else 1.0)
            elapsed = time.time() - t0
            mass_str = ":".join([f"{m:.1f}" for m in masses])
            print(f"  [{label}] Gen {gen:4d}/{ngens}: fit={pf[0].item():+.2f}  r(Fx)={r_fx:.3f}  mass={mass_str}  ({elapsed/60:.1f}min)")

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

    total = (time.time()-t0)/60
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return pop, pf, gen_log, fitness_log, rfx_log, mass_log, total


def main():
    NSTEPS = 600; GAP = 0.5; PSZ = 200
    gx,gy,gz,sp = 10,5,4,0.35
    results = {}

    # ================================================================
    print("="*70)
    print("EXP D: MORPHOLOGICAL ANNEALING (10:1 → 1:1 over 500 gen)")
    print("="*70)
    data2 = build_bodies_2(gx,gy,gz,sp,GAP)

    def mass_anneal(gen):
        """Anneal from 10:1 to 1:1 over first 300 gens, then 1:1 for rest."""
        if gen < 100:
            return [3.0, 0.3]  # Phase 1: force differentiation
        elif gen < 300:
            # Linear interpolation from 10:1 to 1:1
            t = (gen - 100) / 200.0  # 0 to 1
            m0 = 3.0 * (1-t) + 1.0 * t   # 3.0 → 1.0
            m1 = 0.3 * (1-t) + 1.0 * t    # 0.3 → 1.0
            return [m0, m1]
        else:
            return [1.0, 1.0]  # Phase 3: optimize with symmetric body

    pop_d, pf_d, gl_d, fl_d, rl_d, ml_d, t_d = evolve(
        data2, NSTEPS, 500, PSZ, mass_anneal, probe_interval=25, label="anneal")

    # Also run control: symmetric 1:1 for 500 gens
    print("\n--- Control: symmetric 1:1 for 500 gens ---")
    pop_ctrl, pf_ctrl, gl_ctrl, fl_ctrl, rl_ctrl, _, t_ctrl = evolve(
        data2, NSTEPS, 500, PSZ, lambda g: [1.0, 1.0], probe_interval=50, label="control_1:1")

    best_anneal = fl_d[-1]; best_ctrl = fl_ctrl[-1]
    r_anneal = rl_d[-1]; r_ctrl = rl_ctrl[-1]
    results["exp_d_annealing"] = {
        "fitness_annealed": round(best_anneal, 2),
        "fitness_control": round(best_ctrl, 2),
        "improvement": round((best_anneal/best_ctrl - 1)*100, 1) if best_ctrl > 0 else 0,
        "r_fx_annealed": round(r_anneal, 3),
        "r_fx_control": round(r_ctrl, 3),
        "generations": gl_d, "fitness": fl_d, "rfx": rl_d, "mass_ratio": ml_d,
        "control_gens": gl_ctrl, "control_fitness": fl_ctrl,
        "elapsed_min": round(t_d + t_ctrl, 1)
    }
    print(f"\n  RESULT: Annealed={best_anneal:+.2f} vs Control={best_ctrl:+.2f} "
          f"({'ANNEALING WINS!' if best_anneal > best_ctrl else 'Control wins'})")

    # ================================================================
    print("\n" + "="*70)
    print("EXP E: DIFFERENTIATION MEMORY (10:1 → sudden 1:1)")
    print("="*70)

    # Phase 1: Evolve under 10:1 for 250 gens
    print("--- Phase 1: Evolve under 10:1 for 250 gens ---")
    pop_e1, pf_e1, gl_e1, fl_e1, rl_e1, _, t_e1 = evolve(
        data2, NSTEPS, 250, PSZ, lambda g: [3.0, 0.3], probe_interval=50, label="memory_10:1")

    # Phase 2: Switch to 1:1, continue from same population
    print("\n--- Phase 2: Switch to 1:1, continue 250 gens ---")
    pop_e2, pf_e2, gl_e2, fl_e2, rl_e2, _, t_e2 = evolve(
        data2, NSTEPS, 250, PSZ, lambda g: [1.0, 1.0], probe_interval=25, label="memory_1:1",
        init_pop=pop_e1, init_pf=torch.full((PSZ,), float('-inf'), device=DEVICE))
    # Offset gen numbers for phase 2
    gl_e2 = [g + 250 for g in gl_e2]

    r_memory_final = rl_e2[-1]; fit_memory_final = fl_e2[-1]
    results["exp_e_memory"] = {
        "phase1_final_fitness": round(fl_e1[-1], 2),
        "phase1_final_rfx": round(rl_e1[-1], 3),
        "phase2_final_fitness": round(fit_memory_final, 2),
        "phase2_final_rfx": round(r_memory_final, 3),
        "control_fitness": round(best_ctrl, 2),
        "memory_preserved": r_memory_final < r_ctrl - 0.1 if r_ctrl else False,
        "phase1_gens": gl_e1, "phase1_fitness": fl_e1, "phase1_rfx": rl_e1,
        "phase2_gens": gl_e2, "phase2_fitness": fl_e2, "phase2_rfx": rl_e2,
        "elapsed_min": round(t_e1 + t_e2, 1)
    }
    print(f"\n  RESULT: Memory r={r_memory_final:.3f} vs Control r={r_ctrl:.3f} "
          f"({'MEMORY RETAINED!' if r_memory_final < r_ctrl - 0.1 else 'Memory lost'})")

    # ================================================================
    print("\n" + "="*70)
    print("EXP F: 3-BODY EXTENSION")
    print("="*70)

    # Use smaller bodies for 3-body to keep particle count manageable
    gx3,gy3,gz3,sp3 = 7,4,3,0.4
    conditions_3 = [
        ("F0_symmetric", [1.0, 1.0, 1.0]),
        ("F1_partial", [3.0, 0.3, 0.3]),
        ("F2_all_asym", [3.0, 1.0, 0.3]),
    ]
    results["exp_f_3body"] = {}

    for cond_name, masses in conditions_3:
        print(f"\n--- {cond_name}: masses {masses} ---")
        data3 = build_bodies_3(gx3,gy3,gz3,sp3,GAP)
        pop_f, pf_f, gl_f, fl_f, rl_f, _, t_f = evolve(
            data3, NSTEPS, 150, PSZ, lambda g, m=masses: m,
            probe_interval=50, label=cond_name, n_bodies=3)

        # Get r(Fx) for all pairs
        best_genes_f = pop_f[0].cpu().numpy()
        r_01 = replay_rfx_pair(best_genes_f, data3, NSTEPS, masses, 0, 1, n_bodies=3)
        r_02 = replay_rfx_pair(best_genes_f, data3, NSTEPS, masses, 0, 2, n_bodies=3)
        r_12 = replay_rfx_pair(best_genes_f, data3, NSTEPS, masses, 1, 2, n_bodies=3)

        print(f"  r(0-1)={r_01:.3f}, r(0-2)={r_02:.3f}, r(1-2)={r_12:.3f}")

        results["exp_f_3body"][cond_name] = {
            "masses": masses,
            "fitness": round(fl_f[-1], 2),
            "r_01": round(r_01, 3), "r_02": round(r_02, 3), "r_12": round(r_12, 3),
            "elapsed_min": round(t_f, 1)
        }

    # ================================================================
    # SUMMARY FIGURE
    # ================================================================
    print("\nGenerating summary figure...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Season 2: Annealing, Memory, and 3-Body Extension", fontsize=14, fontweight="bold")

    # Panel 1: Morphological Annealing
    ax = axes[0]
    ax.plot(gl_d, fl_d, "o-", color="#e74c3c", markersize=3, linewidth=2, label="Annealed (10:1→1:1)")
    ax.plot(gl_ctrl, fl_ctrl, "s-", color="#95a5a6", markersize=3, linewidth=2, label="Control (1:1 only)")
    ax.axvline(x=100, color="blue", linestyle="--", alpha=0.3, label="Anneal start")
    ax.axvline(x=300, color="green", linestyle="--", alpha=0.3, label="Anneal end (1:1)")
    ax.fill_betweenx([0, max(max(fl_d),max(fl_ctrl))*1.1], 0, 100, alpha=0.05, color="red")
    ax.fill_betweenx([0, max(max(fl_d),max(fl_ctrl))*1.1], 100, 300, alpha=0.05, color="orange")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax.set_title(f"D: Morphological Annealing\nAnnealed={best_anneal:+.1f} vs Control={best_ctrl:+.1f}")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 2: Memory
    ax = axes[1]
    ax.plot(gl_e1, fl_e1, "o-", color="#e74c3c", markersize=3, linewidth=2, label="Phase 1 (10:1)")
    ax.plot(gl_e2, fl_e2, "s-", color="#3498db", markersize=3, linewidth=2, label="Phase 2 (1:1)")
    ax.axvline(x=250, color="black", linestyle="--", alpha=0.5, label="Mass switch")
    ax.axhline(y=best_ctrl, color="#95a5a6", linestyle="--", alpha=0.5, label=f"Control={best_ctrl:+.0f}")
    ax2 = ax.twinx()
    ax2.plot(gl_e1, rl_e1, "^-", color="#2ecc71", markersize=3, linewidth=1, alpha=0.6, label="r(Fx)")
    ax2.plot(gl_e2, rl_e2, "^-", color="#2ecc71", markersize=3, linewidth=1, alpha=0.6)
    ax2.set_ylabel("r(Fx)", color="#2ecc71")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax.set_title(f"E: Differentiation Memory\nfinal r={r_memory_final:.3f}")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)

    # Panel 3: 3-Body
    ax = axes[2]
    cond_labels = ["1:1:1\n(sym)", "10:1:1\n(partial)", "10:3:1\n(all asym)"]
    cond_keys = ["F0_symmetric", "F1_partial", "F2_all_asym"]
    x_pos = np.arange(3)
    r_01_vals = [results["exp_f_3body"][k]["r_01"] for k in cond_keys]
    r_02_vals = [results["exp_f_3body"][k]["r_02"] for k in cond_keys]
    r_12_vals = [results["exp_f_3body"][k]["r_12"] for k in cond_keys]
    width = 0.25
    ax.bar(x_pos-width, r_01_vals, width, color="#e74c3c", alpha=0.8, label="r(0-1)")
    ax.bar(x_pos, r_02_vals, width, color="#3498db", alpha=0.8, label="r(0-2)")
    ax.bar(x_pos+width, r_12_vals, width, color="#2ecc71", alpha=0.8, label="r(1-2)")
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.4, label="Diff threshold")
    ax.set_xticks(x_pos); ax.set_xticklabels(cond_labels)
    ax.set_ylabel("r(Fx)"); ax.set_ylim(-0.5, 1.0)
    ax.set_title("F: 3-Body Symmetry Locks")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "season2_experiments.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")

    # Save JSON
    log_path = os.path.join(RESULTS_DIR, "season2_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Log saved: {log_path}")

    # Beep
    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
    print("\nAll Season 2 experiments complete!")


if __name__ == "__main__":
    main()
