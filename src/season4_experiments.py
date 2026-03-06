"""
Season 4 Experiments: Friction Sweet Spot + Double Asymmetry
============================================================
Exp 11: Friction Sweet Spot - systematic sweep of friction ratios
Exp 10: Body × Environment Double Asymmetry - 3:1 mass + friction asymmetry
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
INPUT_SIZE = 7; HIDDEN_SIZE = 32; OUTPUT_SIZE = 3
N_GENES = INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE + 1
N_W1 = INPUT_SIZE*HIDDEN_SIZE


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
def simulate(genomes, rp, npt, bit, sat, sbt, rlt, N, nper, nsteps,
             per_body_mass=None, friction_map=None):
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
    mass = torch.ones(B,N,device=DEVICE)
    if per_body_mass:
        mass[:, b0m] = per_body_mass[0]; mass[:, b1m] = per_body_mass[1]
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
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1);o=torch.tanh(torch.bmm(h,W2)+b2)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
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


def replay_rfx(genes, data, nsteps, per_body_mass, friction_map=None):
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt
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
    mass=torch.ones(B,N,device=DEVICE)
    if per_body_mass:
        mass[:,b0m]=per_body_mass[0];mass[:,b1m]=per_body_mass[1]
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
        st=torch.full((B,N,1),sin_val,device=DEVICE);ct=torch.full((B,N,1),cos_val,device=DEVICE)
        nn_in=torch.cat([st,ct,ni,bid,comb.expand(B,N,1)],dim=2)
        h=torch.tanh(torch.bmm(nn_in,W1)+b1g);o=torch.tanh(torch.bmm(h,W2)+b2g)
        og=(pos[:,:,1]<GROUND_Y+0.3).float();gc=0.5+og
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
        f[:,:,0]-=fric.unsqueeze(0)*vel[:,:,0]*bl;f[:,:,2]-=fric.unsqueeze(0)*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext
        inv_mass=1.0/mass.clamp(min=0.01);acc=f*inv_mass.unsqueeze(2)
        vel+=acc*DT;vel.clamp_(-50,50);pos+=vel*DT
    if np.std(fx0_list)<1e-10 or np.std(fx1_list)<1e-10: return 0.0
    r_fx,_=pearsonr(fx0_list,fx1_list)
    return r_fx


def evolve(data, nsteps, ngens, psz, per_body_mass, label, friction_map=None):
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt
    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
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
            f,_=simulate(pop[ix],rp,npt,bit,sat,sbt,rlt,N,nper,nsteps,per_body_mass,friction_map)
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
    NSTEPS=600;GAP=0.5;PSZ=200;NGENS=150
    gx,gy,gz,sp=10,5,4,0.35
    data=build_bodies(gx,gy,gz,sp,GAP)
    results={}

    # ================================================================
    print("="*70)
    print("EXP 11: FRICTION SWEET SPOT")
    print("="*70)
    # Friction ratios to test (body0:body1)
    # Keep geometric mean roughly constant: sqrt(f0*f1) ~ sqrt(3*3) = 3
    friction_conditions = [
        (3.0, 3.0, "1:1"),
        (2.0, 4.5, "1:2.25"),
        (1.0, 5.0, "1:5"),
        (0.5, 5.0, "1:10"),
        (0.5, 3.0, "1:6_mild"),
        (0.1, 5.0, "1:50_extreme"),
        (0.1, 10.0, "1:100_ultra"),
    ]

    exp11_results = []
    for f0, f1, ratio_label in friction_conditions:
        fmap = {"body0_friction": f0, "body1_friction": f1}
        label = f"F11_{ratio_label}"
        print(f"\n--- {label}: friction [{f0}, {f1}] ---")
        best_g, gl, fl, elapsed = evolve(data, NSTEPS, NGENS, PSZ, [1.0, 1.0], label, friction_map=fmap)
        r_fx = replay_rfx(best_g, data, NSTEPS, [1.0, 1.0], friction_map=fmap)
        diff_score = 1.0 - abs(r_fx)
        exp11_results.append({
            "f0": f0, "f1": f1, "ratio": round(f1/f0, 1), "label": ratio_label,
            "fitness": round(fl[-1], 2), "r_fx": round(r_fx, 3),
            "diff_score": round(diff_score, 3),
            "diff_x_fit": round(diff_score * fl[-1], 2),
            "elapsed": round(elapsed, 1)
        })
        print(f"  r(Fx)={r_fx:.3f}, diff_score={diff_score:.3f}, fit={fl[-1]:+.2f}")
    results["exp11_friction_sweetspot"] = exp11_results

    # ================================================================
    print("\n" + "="*70)
    print("EXP 10: BODY × ENVIRONMENT DOUBLE ASYMMETRY")
    print("="*70)

    double_conditions = [
        # (mass0, mass1, fric0, fric1, label)
        (1.0, 1.0, 3.0, 3.0, "baseline_sym"),
        (1.0, 1.0, 0.1, 5.0, "env_only"),
        (np.sqrt(3), 1/np.sqrt(3), 3.0, 3.0, "body_only_3:1"),
        (np.sqrt(3), 1/np.sqrt(3), 0.1, 5.0, "double_aligned"),    # heavy on ice, light on grip
        (1/np.sqrt(3), np.sqrt(3), 0.1, 5.0, "double_opposed"),    # light on ice, heavy on grip
        (np.sqrt(10), 1/np.sqrt(10), 0.1, 5.0, "double_10:1"),     # extreme mass + friction
    ]

    exp10_results = []
    for m0, m1, f0, f1, label in double_conditions:
        fmap = {"body0_friction": f0, "body1_friction": f1}
        full_label = f"D10_{label}"
        print(f"\n--- {full_label}: mass [{m0:.2f},{m1:.2f}] friction [{f0},{f1}] ---")
        best_g, gl, fl, elapsed = evolve(data, NSTEPS, NGENS, PSZ, [m0, m1], full_label, friction_map=fmap)
        r_fx = replay_rfx(best_g, data, NSTEPS, [m0, m1], friction_map=fmap)
        exp10_results.append({
            "m0": round(m0,3), "m1": round(m1,3),
            "mass_ratio": round(m0/m1, 1) if m1 > 0 else 999,
            "f0": f0, "f1": f1, "label": label,
            "fitness": round(fl[-1], 2), "r_fx": round(r_fx, 3),
            "elapsed": round(elapsed, 1)
        })
        print(f"  r(Fx)={r_fx:.3f}, fit={fl[-1]:+.2f}")
    results["exp10_double_asymmetry"] = exp10_results

    # ================================================================
    # SUMMARY FIGURE
    # ================================================================
    print("\nGenerating summary figure...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Season 4: Friction Sweet Spot & Double Asymmetry",
                 fontsize=14, fontweight="bold")

    # Panel 1: Friction Sweet Spot
    ax = axes[0]
    ratios = [h["ratio"] for h in exp11_results]
    fits = [h["fitness"] for h in exp11_results]
    rfxs = [h["r_fx"] for h in exp11_results]
    ax.plot(ratios, fits, "o-", color="#e74c3c", linewidth=2, markersize=8, label="Fitness", zorder=3)
    ax2 = ax.twinx()
    ax2.plot(ratios, rfxs, "s-", color="#3498db", linewidth=2, markersize=8, label="r(Fx)", zorder=3)
    ax2.axhline(y=0.3, color="red", linestyle="--", alpha=0.3, label="Diff threshold")
    ax2.axhline(y=0.742, color="gray", linestyle="--", alpha=0.3, label="Sym baseline")
    ax.set_xlabel("Friction Ratio (f1/f0)"); ax.set_ylabel("Fitness", color="#e74c3c")
    ax2.set_ylabel("r(Fx)", color="#3498db"); ax2.set_ylim(-0.5, 1.0)
    ax.set_xscale("log")
    ax.set_title("Exp 11: Friction Sweet Spot")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=7, loc="center left")
    ax.grid(alpha=0.3)
    for i,h in enumerate(exp11_results):
        ax.annotate(f"r={h['r_fx']:.2f}\nfit={h['fitness']:+.0f}",
                     (ratios[i], fits[i]), fontsize=6, ha="center",
                     xytext=(0, 12), textcoords="offset points")

    # Panel 2: Double Asymmetry
    ax = axes[1]
    labels_d = [r["label"].replace("_","\n") for r in exp10_results]
    x_pos = np.arange(len(exp10_results))
    rfx_d = [r["r_fx"] for r in exp10_results]
    fit_d = [r["fitness"] for r in exp10_results]
    colors_d = ["#95a5a6", "#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]
    bars = ax.bar(x_pos - 0.15, rfx_d, 0.3, color=colors_d[:len(exp10_results)], alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar(x_pos + 0.15, fit_d, 0.3, color=colors_d[:len(exp10_results)], alpha=0.35)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.4)
    ax.axhline(y=0.742, color="gray", linestyle="--", alpha=0.4)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels_d, fontsize=7)
    ax.set_ylabel("r(Fx)"); ax2.set_ylabel("Fitness", alpha=0.5)
    ax.set_ylim(-0.5, 1.0)
    for i, r in enumerate(exp10_results):
        ax.text(i, rfx_d[i]+0.05, f"r={rfx_d[i]:.2f}", ha="center", fontsize=7)
        ax.text(i, rfx_d[i]-0.12, f"fit={fit_d[i]:+.0f}", ha="center", fontsize=6, color="gray")
    ax.set_title("Exp 10: Double Asymmetry (Body × Environment)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "season4_experiments.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")

    log_path = os.path.join(RESULTS_DIR, "season4_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Log saved: {log_path}")

    # Final summary
    print("\n" + "="*70)
    print("SEASON 4 SUMMARY")
    print("="*70)
    print("\nExp 11: Friction Sweet Spot")
    for h in exp11_results:
        print(f"  {h['label']:>15s}  ratio={h['ratio']:6.1f}  fit={h['fitness']:+7.2f}  r(Fx)={h['r_fx']:+.3f}  diff×fit={h['diff_x_fit']:+.2f}")
    print(f"\nExp 10: Double Asymmetry")
    for r in exp10_results:
        print(f"  {r['label']:>20s}  mass={r['mass_ratio']:.1f}:1  fit={r['fitness']:+7.2f}  r(Fx)={r['r_fx']:+.3f}")

    # Check for synergy
    env_fit = next(r["fitness"] for r in exp10_results if r["label"] == "env_only")
    body_fit = next(r["fitness"] for r in exp10_results if r["label"] == "body_only_3:1")
    aligned_fit = next(r["fitness"] for r in exp10_results if r["label"] == "double_aligned")
    synergy = aligned_fit > max(env_fit, body_fit)
    print(f"\n  Synergy test: env={env_fit:+.1f}, body={body_fit:+.1f}, double={aligned_fit:+.1f}")
    print(f"  {'SYNERGY!' if synergy else 'No synergy (interference)'}")

    try:
        import winsound
        for _ in range(5): winsound.Beep(800, 300); time.sleep(0.2)
    except: pass
    print("\nAll Season 4 experiments complete!")


if __name__ == "__main__":
    main()
