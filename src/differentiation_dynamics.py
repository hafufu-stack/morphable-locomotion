"""
Differentiation Dynamics: Is role differentiation transient or steady-state?
============================================================================
Track r(Fx) at regular intervals during evolution to observe the temporal
dynamics of functional differentiation under mass asymmetry.

Hypothesis: differentiation peaks early (when the NN first exploits mass
asymmetry) then may be reabsorbed as evolution finds more efficient
synchronized strategies. This would reveal a "U-shaped" differentiation curve.

Setup: 1000 generations, 10:1 mass ratio, replay every 25 gens to measure r(Fx).
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
def simulate(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps, per_body_mass=None):
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
    spp=((sp_-8.0).clamp(min=0)*1.5).sum(dim=1);bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
    c0=pos[:,b0m].mean(dim=1);c1=pos[:,b1m].mean(dim=1)
    coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
    fitness = disp-dz-spp-bw-ep+coh
    fitness = torch.where(torch.isnan(fitness), torch.tensor(-9999.0, device=DEVICE), fitness)
    return fitness, disp


def replay_rfx(genes, data, nsteps, per_body_mass):
    """Quick replay to get r(Fx) only."""
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
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,-1,-1)
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone()
    comb=torch.zeros(B,1,1,device=DEVICE)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False
    mass = torch.ones(N, device=DEVICE)
    mass[b0m] = per_body_mass[0]; mass[b1m] = per_body_mass[1]
    fx0_list, fx1_list = [], []
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
        fx0_list.append(ext[0,b0m,0].mean().item())
        fx1_list.append(ext[0,b1m,0].mean().item())
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
    # Handle constant arrays
    if np.std(fx0_list) < 1e-10 or np.std(fx1_list) < 1e-10:
        return 0.0  # One body outputs constant force = differentiated
    r_fx, _ = pearsonr(fx0_list, fx1_list)
    return r_fx


def main():
    print("="*70)
    print("DIFFERENTIATION DYNAMICS: Transient or Steady-State?")
    print("="*70)

    NSTEPS = 600; GAP = 0.5; PSZ = 200; NGENS = 1000
    PROBE_INTERVAL = 25  # Replay every 25 gens
    PER_BODY_MASS = [3.0, 0.3]  # 10:1 extreme

    gx,gy,gz,sp = 10,5,4,0.35
    data = build_bodies(gx,gy,gz,sp,GAP)
    ap,np_,bi,sa,sb,rl,nper,nt = data
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)

    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    pop=torch.randn(PSZ,N_GENES,device=DEVICE)*0.3
    pop[:,:N_W1]*=s1/0.3;pop[:,N_W1:N_W1+N_B1]=0
    pop[:,N_W1+N_B1:N_W1+N_B1+N_W2]*=s2/0.3
    pop[:,N_W1+N_B1+N_W2:N_W1+N_B1+N_W2+N_B2]=0
    pop[:,-1]=torch.empty(PSZ,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((PSZ,),float('-inf'),device=DEVICE)

    # Tracking arrays
    gen_log = []
    fitness_log = []
    rfx_log = []

    t0 = time.time()
    for gen in range(NGENS):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,_=simulate(pop[ix],rp,npt,bit,sat,sbt,rlt,nt,nper,NSTEPS,PER_BODY_MASS)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]

        # Probe differentiation at intervals
        if gen % PROBE_INTERVAL == 0:
            best_genes = pop[0].cpu().numpy()
            r_fx = replay_rfx(best_genes, data, NSTEPS, PER_BODY_MASS)
            gen_log.append(gen)
            fitness_log.append(pf[0].item())
            rfx_log.append(r_fx)
            elapsed = time.time() - t0
            print(f"  Gen {gen:4d}/{NGENS}: fit={pf[0].item():+.2f}  r(Fx)={r_fx:.3f}  ({elapsed/60:.1f}min)")

        ne=max(2,int(PSZ*0.05));np2=pop[:ne].clone();nf2=pf[:ne].clone()
        nfr=max(2,int(PSZ*0.05));fr=torch.randn(nfr,N_GENES,device=DEVICE)*0.3
        fr[:,:N_W1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np2=torch.cat([np2,fr]);nf2=torch.cat([nf2,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=PSZ-np2.shape[0]
        t1=torch.randint(PSZ,(nc,5),device=DEVICE)
        p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(PSZ,(nc,5),device=DEVICE)
        p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,N_GENES,device=DEVICE)<0.5
        ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,N_GENES,device=DEVICE)<0.15
        ch+=torch.randn(nc,N_GENES,device=DEVICE)*0.3*mt.float()
        np2=torch.cat([np2,ch]);nf2=torch.cat([nf2,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np2;pf=nf2

    total_time = (time.time() - t0) / 60
    print(f"\nTotal time: {total_time:.1f} min")

    # Final probe
    best_genes = pop[0].cpu().numpy()
    r_fx_final = replay_rfx(best_genes, data, NSTEPS, PER_BODY_MASS)
    if gen_log[-1] != NGENS-1:
        gen_log.append(NGENS-1)
        fitness_log.append(pf[0].item())
        rfx_log.append(r_fx_final)

    # ================================================================
    # FIGURE: Differentiation Dynamics Curve
    # ================================================================
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle("Differentiation Dynamics: Mass Asymmetry 10:1 over 1000 Generations",
                 fontsize=14, fontweight="bold")

    color_fit = "#e74c3c"
    color_rfx = "#3498db"

    # Fitness curve
    ax1.plot(gen_log, fitness_log, "o-", color=color_fit, markersize=4, linewidth=2,
             label="Fitness", alpha=0.8)
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Fitness", color=color_fit, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_fit)

    # r(Fx) curve on twin axis
    ax2 = ax1.twinx()
    ax2.plot(gen_log, rfx_log, "s-", color=color_rfx, markersize=4, linewidth=2,
             label="r(Fx)", alpha=0.8)
    ax2.set_ylabel("r(Fx) Force Correlation", color=color_rfx, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_rfx)

    # Reference lines
    ax2.axhline(y=0.742, color="gray", linestyle="--", alpha=0.4, label="Symmetric baseline (0.742)")
    ax2.axhline(y=0.0, color="black", linestyle="-", alpha=0.2)
    ax2.axhline(y=0.3, color="red", linestyle="--", alpha=0.4, label="Differentiation threshold")

    # Phase annotations
    # Find min r(Fx) point
    min_rfx_idx = np.argmin(rfx_log)
    min_rfx_gen = gen_log[min_rfx_idx]
    min_rfx_val = rfx_log[min_rfx_idx]
    ax2.annotate(f"Peak differentiation\nr={min_rfx_val:.3f} @ gen {min_rfx_gen}",
                xy=(min_rfx_gen, min_rfx_val),
                xytext=(min_rfx_gen+100, min_rfx_val+0.2),
                arrowprops=dict(arrowstyle="->", color="blue"),
                fontsize=10, color="blue", fontweight="bold")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="center right", fontsize=9)

    ax1.set_xlim(-10, NGENS+10)
    ax2.set_ylim(-0.5, 1.0)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "differentiation_dynamics.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")

    # Save JSON log
    results = {
        "experiment": "differentiation_dynamics",
        "mass_ratio": "10:1",
        "ngens": NGENS,
        "probe_interval": PROBE_INTERVAL,
        "generations": gen_log,
        "fitness": fitness_log,
        "r_fx": rfx_log,
        "peak_differentiation_gen": int(min_rfx_gen),
        "peak_differentiation_rfx": round(float(min_rfx_val), 3),
        "final_fitness": round(float(fitness_log[-1]), 2),
        "final_rfx": round(float(rfx_log[-1]), 3),
        "total_time_min": round(total_time, 1)
    }
    log_path = os.path.join(RESULTS_DIR, "differentiation_dynamics_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Log saved: {log_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Peak differentiation: r(Fx)={min_rfx_val:.3f} at gen {min_rfx_gen}")
    print(f"Final state: r(Fx)={rfx_log[-1]:.3f}, fitness={fitness_log[-1]:+.2f}")
    if rfx_log[-1] > min_rfx_val + 0.1:
        print("  => TRANSIENT: Differentiation is reabsorbed over time!")
    else:
        print("  => STEADY-STATE: Differentiation is maintained!")

    # Beep
    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
    print("\nDone!")


if __name__ == "__main__":
    main()
