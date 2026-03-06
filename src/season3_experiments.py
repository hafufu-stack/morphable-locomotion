"""
Season 3 Experiments: Sweet Spot, Developmental Unlocking, Swamp Test
=====================================================================
Exp H: Sweet Spot - find optimal mass ratio for differentiation vs fitness
Exp I: Developmental Unlocking - break DoF Trap with curriculum learning
Exp J: Swamp Test - environmental asymmetry with symmetric bodies
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


# ========================================================================
# NN dimensions
# ========================================================================
INPUT_SIZE_STD = 7; INPUT_SIZE_DYN = 8  # +1 for mass input
HIDDEN_SIZE = 32
OUTPUT_SIZE_STD = 3; OUTPUT_SIZE_DYN = 4  # +1 for mass desire

def get_n_genes(input_size, output_size):
    return input_size*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*output_size + output_size + 1


# ========================================================================
# Body building
# ========================================================================
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


# ========================================================================
# Unified simulate function
# ========================================================================
@torch.no_grad()
def simulate(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps,
             per_body_mass=None, dynamic_mass=False, friction_map=None):
    """
    Unified simulation.
    friction_map: if provided, dict with 'body0_friction' and 'body1_friction'
    """
    input_size = INPUT_SIZE_DYN if dynamic_mass else INPUT_SIZE_STD
    output_size = OUTPUT_SIZE_DYN if dynamic_mass else OUTPUT_SIZE_STD
    n_w1 = input_size*HIDDEN_SIZE; n_b1 = HIDDEN_SIZE
    n_w2 = HIDDEN_SIZE*output_size; n_b2 = output_size
    FRICTION_DEFAULT = 3.0

    B = genomes.shape[0]; pos = rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel = torch.zeros(B,N,3,device=DEVICE)
    idx=0; W1=genomes[:,idx:idx+n_w1].reshape(B,input_size,HIDDEN_SIZE);idx+=n_w1
    b1=genomes[:,idx:idx+n_b1].unsqueeze(1);idx+=n_b1
    W2=genomes[:,idx:idx+n_w2].reshape(B,HIDDEN_SIZE,output_size);idx+=n_w2
    b2=genomes[:,idx:idx+n_b2].unsqueeze(1);idx+=n_b2
    freq=genomes[:,idx].abs()
    sx=pos[:,:,0].mean(dim=1);bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False
    mass = torch.ones(B,N,device=DEVICE) if not dynamic_mass else torch.ones(B,N,device=DEVICE)
    if per_body_mass:
        mass[:, b0m] = per_body_mass[0]; mass[:, b1m] = per_body_mass[1]
    total_mass_b0 = mass[:, b0m].sum(dim=1, keepdim=True)
    total_mass_b1 = mass[:, b1m].sum(dim=1, keepdim=True)

    # Friction per particle
    fric = torch.full((N,), FRICTION_DEFAULT, device=DEVICE)
    if friction_map:
        fric[b0m] = friction_map['body0_friction']
        fric[b1m] = friction_map['body1_friction']

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
        if dynamic_mass:
            mass_input = torch.log(mass + 0.1).unsqueeze(2)
            nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1),mass_input],dim=2)
        else:
            nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1);o=torch.tanh(torch.bmm(h,W2)+b2)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))

        if dynamic_mass:
            mass_desire = o[:,:,3]
            alpha_mass = 0.05
            for bm, tm in [(b0m, total_mass_b0), (b1m, total_mass_b1)]:
                d = mass_desire[:, bm]
                w = torch.softmax(d * 3.0, dim=1)
                np_count = bm.sum().item()
                tgt = (w * tm * np_count).clamp(0.1, 10.0)
                tgt = tgt / tgt.sum(dim=1, keepdim=True) * tm * np_count
                mass[:, bm] = (1-alpha_mass) * mass[:, bm] + alpha_mass * tgt

        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d=pb-pa;di=torch.norm(d,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        # Per-particle friction
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


def replay_rfx(genes, data, nsteps, per_body_mass, dynamic_mass=False, friction_map=None):
    """Replay to get r(Fx)."""
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt
    input_size = INPUT_SIZE_DYN if dynamic_mass else INPUT_SIZE_STD
    output_size = OUTPUT_SIZE_DYN if dynamic_mass else OUTPUT_SIZE_STD
    n_w1=input_size*HIDDEN_SIZE;n_b1=HIDDEN_SIZE;n_w2=HIDDEN_SIZE*output_size;n_b2=output_size
    FRICTION_DEFAULT = 3.0
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    genome=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(B,N,3,device=DEVICE)
    gidx=0;W1=genome[:,gidx:gidx+n_w1].reshape(B,input_size,HIDDEN_SIZE);gidx+=n_w1
    b1g=genome[:,gidx:gidx+n_b1].unsqueeze(1);gidx+=n_b1
    W2=genome[:,gidx:gidx+n_w2].reshape(B,HIDDEN_SIZE,output_size);gidx+=n_w2
    b2g=genome[:,gidx:gidx+n_b2].unsqueeze(1);gidx+=n_b2
    freq_val=genome[:,gidx].abs().item()
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False
    mass=torch.ones(B,N,device=DEVICE)
    if per_body_mass:
        mass[:,b0m]=per_body_mass[0];mass[:,b1m]=per_body_mass[1]
    total_mass_b0=mass[:,b0m].sum(dim=1,keepdim=True)
    total_mass_b1=mass[:,b1m].sum(dim=1,keepdim=True)
    fric=torch.full((N,),FRICTION_DEFAULT,device=DEVICE)
    if friction_map:
        fric[b0m]=friction_map['body0_friction'];fric[b1m]=friction_map['body1_friction']
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
        st=torch.full((B,N,1),sin_val,device=DEVICE)
        ct=torch.full((B,N,1),cos_val,device=DEVICE)
        if dynamic_mass:
            mass_input=torch.log(mass+0.1).unsqueeze(2)
            nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1),mass_input],dim=2)
        else:
            nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        fx0_list.append(ext[0,b0m,0].mean().item())
        fx1_list.append(ext[0,b1m,0].mean().item())
        if dynamic_mass:
            mass_desire=o[:,:,3];alpha_mass=0.05
            for bm,tm in [(b0m,total_mass_b0),(b1m,total_mass_b1)]:
                d=mass_desire[:,bm];w=torch.softmax(d*3.0,dim=1)
                np_c=bm.sum().item();tgt=(w*tm*np_c).clamp(0.1,10.0)
                tgt=tgt/tgt.sum(dim=1,keepdim=True)*tm*np_c
                mass[:,bm]=(1-alpha_mass)*mass[:,bm]+alpha_mass*tgt
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
        pa=pos[:,csa];pb=pos[:,csb];d=pb-pa;di=torch.norm(d,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
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


def evolve(data, nsteps, ngens, psz, per_body_mass, label,
           dynamic_mass=False, friction_map=None, init_pop=None):
    """Generic evolution loop."""
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt
    input_size = INPUT_SIZE_DYN if dynamic_mass else INPUT_SIZE_STD
    output_size = OUTPUT_SIZE_DYN if dynamic_mass else OUTPUT_SIZE_STD
    n_genes = get_n_genes(input_size, output_size)
    n_w1=input_size*HIDDEN_SIZE

    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)

    s1=np.sqrt(2.0/(input_size+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+output_size))
    if init_pop is not None:
        pop=init_pop.clone();psz=pop.shape[0]
        pf=torch.full((psz,),float('-inf'),device=DEVICE)
    else:
        pop=torch.randn(psz,n_genes,device=DEVICE)*0.3
        pop[:,:n_w1]*=s1/0.3
        pop[:,n_w1:n_w1+HIDDEN_SIZE]=0
        pop[:,n_w1+HIDDEN_SIZE:n_w1+HIDDEN_SIZE+HIDDEN_SIZE*output_size]*=s2/0.3
        pop[:,n_w1+HIDDEN_SIZE+HIDDEN_SIZE*output_size:n_w1+HIDDEN_SIZE+HIDDEN_SIZE*output_size+output_size]=0
        pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
        pf=torch.full((psz,),float('-inf'),device=DEVICE)

    gen_log,fitness_log=[],[]
    t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,_=simulate(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,
                         per_body_mass,dynamic_mass,friction_map)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]
        if gen%50==0 or gen==ngens-1:
            gen_log.append(gen);fitness_log.append(pf[0].item())
            elapsed=time.time()-t0
            print(f"  [{label}] Gen {gen:4d}/{ngens}: fit={pf[0].item():+.2f}  ({elapsed/60:.1f}min)")
        ne=max(2,int(psz*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,n_genes,device=DEVICE)*0.3
        fr[:,:n_w1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
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
    total=(time.time()-t0)/60
    best_genes=pop[0].cpu().numpy()
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return best_genes, pop, pf, gen_log, fitness_log, total


def main():
    NSTEPS=600;GAP=0.5;PSZ=200
    gx,gy,gz,sp=10,5,4,0.35
    data=build_bodies(gx,gy,gz,sp,GAP)
    results={}

    # ================================================================
    print("="*70)
    print("EXP H: SWEET SPOT - Optimal Mass Ratio for Differentiation")
    print("="*70)
    mass_ratios = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    h_results = []
    for ratio in mass_ratios:
        m0 = np.sqrt(ratio)  # geometric: total mass roughly constant
        m1 = 1.0 / np.sqrt(ratio)
        label = f"H_{ratio:.1f}:1"
        print(f"\n--- {label}: masses [{m0:.2f}, {m1:.2f}] ---")
        best_g, _, _, gl, fl, elapsed = evolve(data, NSTEPS, 150, PSZ, [m0, m1], label)
        r_fx = replay_rfx(best_g, data, NSTEPS, [m0, m1])
        diff_score = 1.0 - abs(r_fx)  # 0=synchronized, 1=differentiated
        efficiency = fl[-1] / max(fl[0], 1)  # fitness improvement ratio
        h_results.append({
            "ratio": ratio, "m0": round(m0,3), "m1": round(m1,3),
            "fitness": round(fl[-1], 2), "r_fx": round(r_fx, 3),
            "diff_score": round(diff_score, 3),
            "diff_x_fit": round(diff_score * fl[-1], 2),
            "elapsed": round(elapsed, 1)
        })
        print(f"  r(Fx)={r_fx:.3f}, diff_score={diff_score:.3f}, fit={fl[-1]:+.2f}")
    results["exp_h_sweetspot"] = h_results

    # ================================================================
    print("\n" + "="*70)
    print("EXP I: DEVELOPMENTAL UNLOCKING (DoF Trap Breakthrough)")
    print("="*70)

    # I0: Control - Fixed 3-output for 500gen
    print("\n--- I0: Control (fixed 3-out, 500gen) ---")
    bg_i0, _, _, gl_i0, fl_i0, t_i0 = evolve(
        data, NSTEPS, 500, PSZ, [1.0, 1.0], "I0_fixed",
        dynamic_mass=False)
    r_i0 = replay_rfx(bg_i0, data, NSTEPS, [1.0, 1.0], dynamic_mass=False)

    # I1: Developmental - 150gen fixed, then unlock 4th output
    # Phase 1: Fixed 3-output for locomotion
    print("\n--- I1: Phase 1 (fixed 3-out, 150gen) ---")
    bg_i1p1, pop_i1p1, pf_i1p1, gl_i1p1, fl_i1p1, t_i1p1 = evolve(
        data, NSTEPS, 150, PSZ, [1.0, 1.0], "I1_phase1_fixed",
        dynamic_mass=False)

    # Phase 2: Expand genome to 4-output, copy weights, unlock mass control
    print("\n--- I1: Phase 2 (unlock 4th output, 350gen) ---")
    n_genes_std = get_n_genes(INPUT_SIZE_STD, OUTPUT_SIZE_STD)
    n_genes_dyn = get_n_genes(INPUT_SIZE_DYN, OUTPUT_SIZE_DYN)
    n_w1_std = INPUT_SIZE_STD*HIDDEN_SIZE
    n_w1_dyn = INPUT_SIZE_DYN*HIDDEN_SIZE

    # Create expanded population
    expanded_pop = torch.randn(PSZ, n_genes_dyn, device=DEVICE) * 0.01
    for i in range(PSZ):
        old = pop_i1p1[i]
        # Copy W1 weights (7x32 -> 8x32): zero-pad input dim
        old_w1 = old[:n_w1_std].reshape(INPUT_SIZE_STD, HIDDEN_SIZE)
        expanded_pop[i, :n_w1_dyn] = 0.0
        exp_w1 = expanded_pop[i, :n_w1_dyn].reshape(INPUT_SIZE_DYN, HIDDEN_SIZE)
        exp_w1[:INPUT_SIZE_STD, :] = old_w1  # copy old weights
        exp_w1[INPUT_SIZE_STD:, :] = 0.0  # new input dim starts at 0
        expanded_pop[i, :n_w1_dyn] = exp_w1.reshape(-1)
        # Copy b1
        old_offset = n_w1_std
        new_offset = n_w1_dyn
        expanded_pop[i, new_offset:new_offset+HIDDEN_SIZE] = old[old_offset:old_offset+HIDDEN_SIZE]
        old_offset += HIDDEN_SIZE; new_offset += HIDDEN_SIZE
        # Copy W2 (32x3 -> 32x4): zero-pad output dim
        old_w2 = old[old_offset:old_offset+HIDDEN_SIZE*OUTPUT_SIZE_STD].reshape(HIDDEN_SIZE, OUTPUT_SIZE_STD)
        exp_w2 = torch.zeros(HIDDEN_SIZE, OUTPUT_SIZE_DYN, device=DEVICE)
        exp_w2[:, :OUTPUT_SIZE_STD] = old_w2
        expanded_pop[i, new_offset:new_offset+HIDDEN_SIZE*OUTPUT_SIZE_DYN] = exp_w2.reshape(-1)
        old_offset += HIDDEN_SIZE*OUTPUT_SIZE_STD; new_offset += HIDDEN_SIZE*OUTPUT_SIZE_DYN
        # Copy b2 (3 -> 4)
        expanded_pop[i, new_offset:new_offset+OUTPUT_SIZE_STD] = old[old_offset:old_offset+OUTPUT_SIZE_STD]
        expanded_pop[i, new_offset+OUTPUT_SIZE_STD:new_offset+OUTPUT_SIZE_DYN] = 0.0
        old_offset += OUTPUT_SIZE_STD; new_offset += OUTPUT_SIZE_DYN
        # Copy freq
        expanded_pop[i, new_offset] = old[old_offset]

    bg_i1p2, _, _, gl_i1p2, fl_i1p2, t_i1p2 = evolve(
        data, NSTEPS, 350, PSZ, [1.0, 1.0], "I1_phase2_unlock",
        dynamic_mass=True, init_pop=expanded_pop)
    gl_i1p2 = [g+150 for g in gl_i1p2]
    r_i1 = replay_rfx(bg_i1p2, data, NSTEPS, [1.0, 1.0], dynamic_mass=True)

    results["exp_i_developmental"] = {
        "i0_fixed_fitness": round(fl_i0[-1], 2),
        "i0_fixed_rfx": round(r_i0, 3),
        "i1_phase1_fitness": round(fl_i1p1[-1], 2),
        "i1_phase2_fitness": round(fl_i1p2[-1], 2),
        "i1_final_rfx": round(r_i1, 3),
        "breakthrough": fl_i1p2[-1] > fl_i0[-1],
        "i0_gens": gl_i0, "i0_fitness": fl_i0,
        "i1_p1_gens": gl_i1p1, "i1_p1_fitness": fl_i1p1,
        "i1_p2_gens": gl_i1p2, "i1_p2_fitness": fl_i1p2,
        "elapsed_min": round(t_i0 + t_i1p1 + t_i1p2, 1)
    }
    print(f"\n  I0 Fixed: fit={fl_i0[-1]:+.2f}")
    print(f"  I1 Developmental: fit={fl_i1p2[-1]:+.2f}")
    print(f"  {'BREAKTHROUGH!' if fl_i1p2[-1] > fl_i0[-1] else 'No breakthrough'}")

    # ================================================================
    print("\n" + "="*70)
    print("EXP J: SWAMP TEST - Environmental Asymmetry")
    print("="*70)

    swamp_conditions = [
        ("J0_uniform", {"body0_friction": 3.0, "body1_friction": 3.0}),
        ("J1_mild", {"body0_friction": 0.5, "body1_friction": 3.0}),
        ("J2_extreme", {"body0_friction": 0.1, "body1_friction": 5.0}),
    ]
    results["exp_j_swamp"] = {}
    for cond_name, fmap in swamp_conditions:
        print(f"\n--- {cond_name}: friction {fmap} ---")
        bg_j, _, _, gl_j, fl_j, t_j = evolve(
            data, NSTEPS, 150, PSZ, [1.0, 1.0], cond_name,
            friction_map=fmap)
        r_j = replay_rfx(bg_j, data, NSTEPS, [1.0, 1.0], friction_map=fmap)
        results["exp_j_swamp"][cond_name] = {
            "friction": fmap, "fitness": round(fl_j[-1], 2),
            "r_fx": round(r_j, 3), "elapsed": round(t_j, 1),
            "gens": gl_j, "fitness_log": fl_j
        }
        print(f"  r(Fx)={r_j:.3f}, fit={fl_j[-1]:+.2f}")

    # ================================================================
    # SUMMARY FIGURE
    # ================================================================
    print("\nGenerating summary figure...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Season 3: Sweet Spot, Developmental Unlocking, Swamp Test",
                 fontsize=14, fontweight="bold")

    # Panel 1: Sweet Spot
    ax = axes[0]
    ratios = [h["ratio"] for h in h_results]
    fits = [h["fitness"] for h in h_results]
    rfxs = [h["r_fx"] for h in h_results]
    dxf = [h["diff_x_fit"] for h in h_results]
    ax.plot(ratios, fits, "o-", color="#e74c3c", linewidth=2, markersize=6, label="Fitness")
    ax2 = ax.twinx()
    ax2.plot(ratios, rfxs, "s-", color="#3498db", linewidth=2, markersize=6, label="r(Fx)")
    ax2.axhline(y=0.3, color="red", linestyle="--", alpha=0.3, label="Diff threshold")
    ax.set_xlabel("Mass Ratio"); ax.set_ylabel("Fitness", color="#e74c3c")
    ax2.set_ylabel("r(Fx)", color="#3498db"); ax2.set_ylim(-0.5, 1.0)
    ax.set_title("H: Sweet Spot (Fitness vs Differentiation)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=7, loc="center right")
    ax.grid(alpha=0.3)

    # Panel 2: Developmental Unlocking
    ax = axes[1]
    ax.plot(gl_i0, fl_i0, "o-", color="#95a5a6", markersize=3, linewidth=2,
            label=f"I0 Fixed: {fl_i0[-1]:+.0f}")
    ax.plot(gl_i1p1, fl_i1p1, "o-", color="#e74c3c", markersize=3, linewidth=2,
            label=f"I1 Phase 1 (fixed)")
    ax.plot(gl_i1p2, fl_i1p2, "s-", color="#3498db", markersize=3, linewidth=2,
            label=f"I1 Phase 2 (unlock): {fl_i1p2[-1]:+.0f}")
    ax.axvline(x=150, color="black", linestyle="--", alpha=0.5, label="Unlock point")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax.set_title("I: Developmental Unlocking")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 3: Swamp Test
    ax = axes[2]
    cond_names = list(results["exp_j_swamp"].keys())
    x_pos = np.arange(len(cond_names))
    rfx_vals = [results["exp_j_swamp"][c]["r_fx"] for c in cond_names]
    fit_vals = [results["exp_j_swamp"][c]["fitness"] for c in cond_names]
    colors_j = ["#95a5a6", "#3498db", "#e74c3c"]
    bars = ax.bar(x_pos-0.15, rfx_vals, 0.3, color=colors_j, alpha=0.8, label="r(Fx)")
    ax2 = ax.twinx()
    ax2.bar(x_pos+0.15, fit_vals, 0.3, color=colors_j, alpha=0.4)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.4)
    ax.axhline(y=0.742, color="gray", linestyle="--", alpha=0.4, label="Sym baseline")
    labels_j = ["Uniform\n3:3", "Mild\n0.5:3", "Extreme\n0.1:5"]
    ax.set_xticks(x_pos); ax.set_xticklabels(labels_j)
    ax.set_ylabel("r(Fx)"); ax2.set_ylabel("Fitness", alpha=0.5)
    ax.set_ylim(-0.5, 1.0)
    for i, c in enumerate(cond_names):
        ax.text(i, rfx_vals[i]+0.05, f"r={rfx_vals[i]:.2f}", ha="center", fontsize=8)
        ax.text(i, rfx_vals[i]-0.12, f"fit={fit_vals[i]:+.0f}", ha="center", fontsize=7, color="gray")
    ax.set_title("J: Swamp Test (Environmental Asymmetry)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "season3_experiments.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")

    # Save JSON
    log_path = os.path.join(RESULTS_DIR, "season3_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Log saved: {log_path}")

    # Final summary
    print("\n" + "="*70)
    print("SEASON 3 SUMMARY")
    print("="*70)
    print("\nH: Sweet Spot")
    for h in h_results:
        print(f"  {h['ratio']:5.1f}:1  fit={h['fitness']:+7.2f}  r(Fx)={h['r_fx']:+.3f}  diff×fit={h['diff_x_fit']:+.2f}")
    print(f"\nI: Developmental Unlocking")
    print(f"  I0 Fixed 500gen: {fl_i0[-1]:+.2f}")
    print(f"  I1 Developmental:  {fl_i1p2[-1]:+.2f}  {'BREAKTHROUGH!' if fl_i1p2[-1]>fl_i0[-1] else 'no breakthrough'}")
    print(f"\nJ: Swamp Test")
    for c in cond_names:
        r = results["exp_j_swamp"][c]
        print(f"  {c}: r(Fx)={r['r_fx']:.3f}  fit={r['fitness']:+.2f}")
    env_diff = results["exp_j_swamp"]["J2_extreme"]["r_fx"] < results["exp_j_swamp"]["J0_uniform"]["r_fx"] - 0.1
    print(f"  Environmental differentiation: {'YES!' if env_diff else 'No'}")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
    print("\nAll Season 3 experiments complete!")


if __name__ == "__main__":
    main()
