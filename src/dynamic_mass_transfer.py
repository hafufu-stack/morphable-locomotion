"""
Experiment G: Dynamic Mass Transfer (Morphable Mode)
=====================================================
Each particle's mass can shift during locomotion. The NN gets a 4th output:
"mass desire" - how much mass this particle wants. Mass is redistributed
per-body each step while conserving total mass.

Conditions:
- G0: Fixed mass 1:1 (control, 150gen)
- G1: Dynamic mass 1:1 (NN controls mass distribution, 150gen)
- G2: Dynamic mass 10:1 (asymmetric + dynamic, 150gen)

Question: Can the NN learn to dynamically shift mass between bodies?
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
INPUT_SIZE = 8  # +1 for current particle mass
HIDDEN_SIZE = 32
OUTPUT_SIZE_FIXED = 3  # fx, fy, fz
OUTPUT_SIZE_DYNAMIC = 4  # fx, fy, fz, mass_desire


def get_n_genes(output_size):
    n_w1 = INPUT_SIZE * HIDDEN_SIZE
    n_b1 = HIDDEN_SIZE
    n_w2 = HIDDEN_SIZE * output_size
    n_b2 = output_size
    return n_w1 + n_b1 + n_w2 + n_b2 + 1  # +1 for freq


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
def simulate_dynamic(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps,
                     per_body_mass=None, dynamic_mass=False):
    """Simulate with optional dynamic mass transfer."""
    out_size = OUTPUT_SIZE_DYNAMIC if dynamic_mass else OUTPUT_SIZE_FIXED
    n_w1 = INPUT_SIZE * HIDDEN_SIZE; n_b1 = HIDDEN_SIZE
    n_w2 = HIDDEN_SIZE * out_size; n_b2 = out_size

    B = genomes.shape[0]; pos = rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel = torch.zeros(B,N,3,device=DEVICE)
    idx=0; W1=genomes[:,idx:idx+n_w1].reshape(B,INPUT_SIZE,HIDDEN_SIZE);idx+=n_w1
    b1=genomes[:,idx:idx+n_b1].unsqueeze(1);idx+=n_b1
    W2=genomes[:,idx:idx+n_w2].reshape(B,HIDDEN_SIZE,out_size);idx+=n_w2
    b2=genomes[:,idx:idx+n_b2].unsqueeze(1);idx+=n_b2
    freq=genomes[:,idx].abs()
    sx=pos[:,:,0].mean(dim=1);bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE);te=torch.zeros(B,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False

    # Initialize per-particle mass
    mass = torch.ones(B,N,device=DEVICE)
    if per_body_mass:
        mass[:, b0m] = per_body_mass[0]; mass[:, b1m] = per_body_mass[1]
    total_mass_b0 = mass[:, b0m].sum(dim=1, keepdim=True)  # [B,1]
    total_mass_b1 = mass[:, b1m].sum(dim=1, keepdim=True)

    # Track mass distribution stats
    mass_std_log = []  # Track mass variance within body 0

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

        # Normalized mass for NN input (per-particle, log scale)
        mass_input = torch.log(mass + 0.1).unsqueeze(2)  # [B,N,1]

        st=torch.sin(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        ct=torch.cos(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1),mass_input],dim=2)  # 8 inputs
        h=torch.tanh(torch.bmm(nn_in,W1)+b1);o=torch.tanh(torch.bmm(h,W2)+b2)

        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5;te+=(ext**2).sum(dim=(1,2))

        # Dynamic mass redistribution
        if dynamic_mass:
            mass_desire = o[:,:,3]  # [B,N], range [-1, 1]
            # Softmax within each body to get mass weights (conserves total mass)
            desire_b0 = mass_desire[:, b0m]  # [B, nper]
            desire_b1 = mass_desire[:, b1m]
            # Smooth transition: blend current mass with desired (slow update)
            alpha_mass = 0.05  # slow mass shift rate
            weights_b0 = torch.softmax(desire_b0 * 3.0, dim=1)  # temperature=3
            weights_b1 = torch.softmax(desire_b1 * 3.0, dim=1)
            target_mass_b0 = weights_b0 * total_mass_b0 * nper  # redistribute
            target_mass_b1 = weights_b1 * total_mass_b1 * nper
            # Clamp to reasonable range
            target_mass_b0 = target_mass_b0.clamp(0.1, 10.0)
            target_mass_b1 = target_mass_b1.clamp(0.1, 10.0)
            # Renormalize to conserve
            target_mass_b0 = target_mass_b0 / target_mass_b0.sum(dim=1, keepdim=True) * total_mass_b0 * nper
            target_mass_b1 = target_mass_b1 / target_mass_b1.sum(dim=1, keepdim=True) * total_mass_b1 * nper
            # Exponential moving average update
            mass[:, b0m] = (1-alpha_mass) * mass[:, b0m] + alpha_mass * target_mass_b0
            mass[:, b1m] = (1-alpha_mass) * mass[:, b1m] + alpha_mass * target_mass_b1

            if step % 100 == 0:
                mass_std_log.append(mass[0, b0m].std().item())

        f=torch.zeros(B,N,3,device=DEVICE)
        f[:,:,1]+=GRAVITY*mass
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

    # Return final mass distribution for analysis
    return fitness, disp, mass, mass_std_log


def replay_analysis(genes, data, nsteps, per_body_mass, dynamic_mass):
    """Replay best genome and collect detailed mass + force data."""
    ap, np_, bi, sa, sb, rl, nper, nt = data
    N = nt
    out_size = OUTPUT_SIZE_DYNAMIC if dynamic_mass else OUTPUT_SIZE_FIXED
    n_w1 = INPUT_SIZE * HIDDEN_SIZE; n_b1 = HIDDEN_SIZE
    n_w2 = HIDDEN_SIZE * out_size; n_b2 = out_size

    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    genome = torch.tensor(genes, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    B=1; pos=rp.unsqueeze(0).clone(); vel=torch.zeros(B,N,3,device=DEVICE)
    gidx=0; W1=genome[:,gidx:gidx+n_w1].reshape(B,INPUT_SIZE,HIDDEN_SIZE);gidx+=n_w1
    b1g=genome[:,gidx:gidx+n_b1].unsqueeze(1);gidx+=n_b1
    W2=genome[:,gidx:gidx+n_w2].reshape(B,HIDDEN_SIZE,out_size);gidx+=n_w2
    b2g=genome[:,gidx:gidx+n_b2].unsqueeze(1);gidx+=n_b2
    freq_val=genome[:,gidx].abs().item()
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False
    mass = torch.ones(B,N,device=DEVICE)
    if per_body_mass:
        mass[:, b0m] = per_body_mass[0]; mass[:, b1m] = per_body_mass[1]
    total_mass_b0 = mass[:, b0m].sum(dim=1, keepdim=True)
    total_mass_b1 = mass[:, b1m].sum(dim=1, keepdim=True)
    nper = (b0m.sum()).item()

    fx0_list, fx1_list = [], []
    mass_max_log, mass_min_log, mass_std_log = [], [], []
    mass_com_x_log = []  # center of mass x of body 0

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

        mass_input = torch.log(mass + 0.1).unsqueeze(2)
        sin_val = np.sin(2*np.pi*freq_val*t)
        cos_val = np.cos(2*np.pi*freq_val*t)
        st=torch.full((B,N,1), sin_val, device=DEVICE)
        ct=torch.full((B,N,1), cos_val, device=DEVICE)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1),mass_input],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
        ext=torch.zeros(B,N,3,device=DEVICE)
        ext[:,:,0]=BASE_AMP*o[:,:,0]*gc;ext[:,:,1]=BASE_AMP*torch.clamp(o[:,:,1],min=0)*gc
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        fx0_list.append(ext[0,b0m,0].mean().item())
        fx1_list.append(ext[0,b1m,0].mean().item())

        if dynamic_mass:
            mass_desire = o[:,:,3]
            desire_b0 = mass_desire[:, b0m]
            desire_b1 = mass_desire[:, b1m]
            alpha_mass = 0.05
            weights_b0 = torch.softmax(desire_b0 * 3.0, dim=1)
            weights_b1 = torch.softmax(desire_b1 * 3.0, dim=1)
            target_mass_b0 = weights_b0 * total_mass_b0 * nper
            target_mass_b1 = weights_b1 * total_mass_b1 * nper
            target_mass_b0 = target_mass_b0.clamp(0.1, 10.0)
            target_mass_b1 = target_mass_b1.clamp(0.1, 10.0)
            target_mass_b0 = target_mass_b0 / target_mass_b0.sum(dim=1, keepdim=True) * total_mass_b0 * nper
            target_mass_b1 = target_mass_b1 / target_mass_b1.sum(dim=1, keepdim=True) * total_mass_b1 * nper
            mass[:, b0m] = (1-alpha_mass) * mass[:, b0m] + alpha_mass * target_mass_b0
            mass[:, b1m] = (1-alpha_mass) * mass[:, b1m] + alpha_mass * target_mass_b1

        # Log mass distribution stats for body 0
        m0 = mass[0, b0m]
        mass_max_log.append(m0.max().item())
        mass_min_log.append(m0.min().item())
        mass_std_log.append(m0.std().item())
        # Weighted center of mass (x-coordinate)
        weighted_x = (pos[0, b0m, 0] * m0).sum() / m0.sum()
        mass_com_x_log.append(weighted_x.item())

        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY*mass
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
        inv_mass = 1.0 / mass.clamp(min=0.01)
        acc = f * inv_mass.unsqueeze(2)
        vel+=acc*DT; vel.clamp_(-50, 50); pos+=vel*DT

    if np.std(fx0_list) < 1e-10 or np.std(fx1_list) < 1e-10:
        r_fx = 0.0
    else:
        r_fx, _ = pearsonr(fx0_list, fx1_list)

    # Mass distribution ratio: max/min within body 0
    final_mass_ratio = mass[0,b0m].max().item() / max(mass[0,b0m].min().item(), 0.01)

    return {
        "r_fx": r_fx,
        "mass_max": mass_max_log,
        "mass_min": mass_min_log,
        "mass_std": mass_std_log,
        "mass_com_x": mass_com_x_log,
        "final_mass_ratio": final_mass_ratio,
        "final_mass_b0": mass[0,b0m].cpu().numpy().tolist(),
    }


def evolve_condition(data, nsteps, ngens, psz, per_body_mass, dynamic_mass, label):
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt
    out_size = OUTPUT_SIZE_DYNAMIC if dynamic_mass else OUTPUT_SIZE_FIXED
    n_genes = get_n_genes(out_size)
    n_w1 = INPUT_SIZE * HIDDEN_SIZE; n_b1 = HIDDEN_SIZE
    n_w2 = HIDDEN_SIZE * out_size; n_b2 = out_size

    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)

    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+out_size))
    pop=torch.randn(psz,n_genes,device=DEVICE)*0.3
    pop[:,:n_w1]*=s1/0.3;pop[:,n_w1:n_w1+n_b1]=0
    pop[:,n_w1+n_b1:n_w1+n_b1+n_w2]*=s2/0.3
    pop[:,n_w1+n_b1+n_w2:n_w1+n_b1+n_w2+n_b2]=0
    pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE)

    gen_log, fitness_log = [], []
    t0 = time.time()

    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,_,_,_=simulate_dynamic(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,
                                      per_body_mass,dynamic_mass)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]

        if gen % 50 == 0 or gen == ngens-1:
            gen_log.append(gen); fitness_log.append(pf[0].item())
            elapsed = time.time() - t0
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

    total = (time.time()-t0)/60
    best_genes = pop[0].cpu().numpy()
    print(f"  [{label}] Done: {total:.1f}min | Best={pf[0].item():+.2f}")
    return best_genes, gen_log, fitness_log, total


def main():
    NSTEPS = 600; GAP = 0.5; PSZ = 200; NGENS = 500
    gx,gy,gz,sp = 10,5,4,0.35
    data = build_bodies(gx,gy,gz,sp,GAP)
    results = {}

    conditions = [
        ("G0_fixed_1:1", [1.0, 1.0], False),
        ("G1_dynamic_1:1", [1.0, 1.0], True),
        ("G2_dynamic_10:1", [3.0, 0.3], True),
    ]

    all_analyses = {}
    for cond_name, masses, dynamic in conditions:
        print(f"\n{'='*70}")
        print(f"  {cond_name}: masses={masses}, dynamic={dynamic}")
        print(f"{'='*70}")

        best_genes, gl, fl, elapsed = evolve_condition(
            data, NSTEPS, NGENS, PSZ, masses, dynamic, cond_name)

        # Detailed replay analysis
        analysis = replay_analysis(best_genes, data, NSTEPS, masses, dynamic)
        all_analyses[cond_name] = analysis

        results[cond_name] = {
            "masses": masses,
            "dynamic": dynamic,
            "fitness": round(fl[-1], 2),
            "r_fx": round(analysis["r_fx"], 3),
            "final_mass_ratio": round(analysis["final_mass_ratio"], 2),
            "mass_std_final": round(analysis["mass_std"][-1], 4) if analysis["mass_std"] else 0,
            "elapsed_min": round(elapsed, 1),
            "generations": gl, "fitness_log": fl
        }
        print(f"  r(Fx)={analysis['r_fx']:.3f}, mass_ratio={analysis['final_mass_ratio']:.2f}")

    # ================================================================
    # FIGURE
    # ================================================================
    print("\nGenerating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Experiment G: Dynamic Mass Transfer (Morphable Mode)", fontsize=14, fontweight="bold")

    # Panel 1: Fitness comparison
    ax = axes[0]
    colors = {"G0_fixed_1:1": "#95a5a6", "G1_dynamic_1:1": "#3498db", "G2_dynamic_10:1": "#e74c3c"}
    labels_clean = {"G0_fixed_1:1": "Fixed 1:1", "G1_dynamic_1:1": "Dynamic 1:1", "G2_dynamic_10:1": "Dynamic 10:1"}
    for cname in results:
        r = results[cname]
        ax.plot(r["generations"], r["fitness_log"], "o-", color=colors[cname],
                markersize=3, linewidth=2, label=f"{labels_clean[cname]}: {r['fitness']:+.0f}")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax.set_title("Fitness Comparison")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Mass distribution over time (Body 0, dynamic conditions)
    ax = axes[1]
    for cname in ["G1_dynamic_1:1", "G2_dynamic_10:1"]:
        a = all_analyses[cname]
        steps = np.arange(len(a["mass_std"]))
        ax.plot(steps, a["mass_std"], "-", color=colors[cname], linewidth=2,
                label=f"{labels_clean[cname]}")
    ax.set_xlabel("Simulation Step"); ax.set_ylabel("Mass Std Dev (Body 0)")
    ax.set_title("Mass Distribution Variance")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3: r(Fx) bar chart
    ax = axes[2]
    cnames = list(results.keys())
    x_pos = np.arange(len(cnames))
    rfx_vals = [results[c]["r_fx"] for c in cnames]
    bars = ax.bar(x_pos, rfx_vals, color=[colors[c] for c in cnames], alpha=0.8)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.4, label="Diff threshold")
    ax.axhline(y=0.742, color="gray", linestyle="--", alpha=0.4, label="Symmetric baseline")
    ax.set_xticks(x_pos); ax.set_xticklabels([labels_clean[c] for c in cnames], fontsize=9)
    ax.set_ylabel("r(Fx)"); ax.set_ylim(-0.5, 1.0)
    # Add mass ratio annotation
    for i, c in enumerate(cnames):
        mr = results[c]["final_mass_ratio"]
        ax.text(i, rfx_vals[i]+0.05, f"ratio={mr:.1f}x", ha="center", fontsize=8, color="black")
    ax.set_title("Force Correlation & Mass Ratio")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "dynamic_mass_transfer.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")

    # Save JSON
    log_path = os.path.join(RESULTS_DIR, "dynamic_mass_log.json")
    # Remove numpy arrays for JSON serialization
    json_results = {}
    for k, v in results.items():
        json_results[k] = {kk: vv for kk, vv in v.items()}
    with open(log_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Log saved: {log_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Dynamic Mass Transfer")
    print("="*70)
    for c in results:
        r = results[c]
        print(f"  {c}: fit={r['fitness']:+.2f}  r(Fx)={r['r_fx']:.3f}  "
              f"mass_ratio={r['final_mass_ratio']:.1f}x")

    dynamic_benefit = results["G1_dynamic_1:1"]["fitness"] - results["G0_fixed_1:1"]["fitness"]
    print(f"\n  Dynamic mass benefit (1:1): {dynamic_benefit:+.2f}")
    if dynamic_benefit > 0:
        print("  => DYNAMIC MASS HELPS! Morphable mode improves locomotion!")
    else:
        print("  => Dynamic mass does not help (or slightly hurts).")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
    print("\nDone!")


if __name__ == "__main__":
    main()
