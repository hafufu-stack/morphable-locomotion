"""
Asymmetric Body Force Analysis: Does Mass Asymmetry Force Role Differentiation?
================================================================================
Hypothesis: Symmetric bodies don't differentiate (r=0.79 confirmed).
           Will asymmetric mass (2.0 vs 0.5) force the NN to assign different roles?

Experiments:
  A. Control: Symmetric bodies (mass=1.0 each) - baseline
  B. Bone+Muscle: Asymmetric (mass=2.0 vs 0.5) - test condition
  C. Extreme: (mass=3.0 vs 0.3) - extreme test

For each condition:
  - Evolve 150 gens
  - Replay with per-body force tracking
  - Compute force correlation (r) between bodies
  - Generate comparison figure

Autonomous: runs to completion, beeps when done.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
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
    # Mass per particle
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
        vel+=acc*DT;pos+=vel*DT
    disp=pos[:,:,0].mean(dim=1)-sx;dz=pos[:,:,2].mean(dim=1).abs()
    sp_=pos.max(dim=1).values-pos.min(dim=1).values
    spp=((sp_-8.0).clamp(min=0)*1.5).sum(dim=1);bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
    c0=pos[:,b0m].mean(dim=1);c1=pos[:,b1m].mean(dim=1)
    coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
    return disp-dz-spp-bw-ep+coh, disp


def evolve(nsteps, gap, ngens, psz, per_body_mass=None, label=""):
    gx,gy,gz,sp = 10,5,4,0.35; data = build_bodies(gx,gy,gz,sp,gap)
    ap,np_,bi,sa,sb,rl,nper,nt = data
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    pop=torch.randn(psz,N_GENES,device=DEVICE)*0.3
    pop[:,:N_W1]*=s1/0.3;pop[:,N_W1:N_W1+N_B1]=0
    pop[:,N_W1+N_B1:N_W1+N_B1+N_W2]*=s2/0.3
    pop[:,N_W1+N_B1+N_W2:N_W1+N_B1+N_W2+N_B2]=0
    pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE); t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,_=simulate(pop[ix],rp,npt,bit,sat,sbt,rlt,nt,nper,nsteps,per_body_mass)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]
        if gen%50==0: print(f"  [{label}] Gen {gen:3d}/{ngens}: best={pf[0].item():+.2f}")
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
    elapsed=time.time()-t0
    print(f"  [{label}] Done: {elapsed/60:.1f}min | Best={pf[0].item():+.2f}")
    return pop[0].cpu().numpy(), pf[0].item(), data


@torch.no_grad()
def replay_forces(genes, data, nsteps, per_body_mass=None):
    ap,np_,bi,sa,sb,rl,nper,nt = data; N=nt
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    npt=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bit=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sat=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sbt=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rlt=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    g=torch.tensor(genes,dtype=torch.float32,device=DEVICE).unsqueeze(0)
    B=1;pos=rp.unsqueeze(0).clone();vel=torch.zeros(1,N,3,device=DEVICE)
    idx=0;W1=g[:,idx:idx+N_W1].reshape(B,INPUT_SIZE,HIDDEN_SIZE);idx+=N_W1
    b1=g[:,idx:idx+N_B1].unsqueeze(1);idx+=N_B1
    W2=g[:,idx:idx+N_W2].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE);idx+=N_W2
    b2=g[:,idx:idx+N_B2].unsqueeze(1);idx+=N_B2
    freq=g[:,idx].abs()
    bid=bit.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=npt.unsqueeze(0).expand(B,N,3)
    b0m=bit==0;b1m=bit==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    csa=sat.clone();csb=sbt.clone();crl=rlt.clone();comb=torch.zeros(B,1,1,device=DEVICE);cd=False
    n_orig=len(sa)
    mass=torch.ones(N,device=DEVICE)
    if per_body_mass: mass[b0m]=per_body_mass[0];mass[b1m]=per_body_mass[1]
    inv_mass=1.0/mass
    
    b0fx,b0fy,b0fz,b1fx,b1fy,b1fz=[],[],[],[],[],[]
    com0x,com1x,cdist_h,spring_h=[],[],[],[]
    tframes,fframes=[],[]
    
    for step in range(nsteps):
        t=step*DT
        c0=pos[0,b0m].mean(dim=0);c1=pos[0,b1m].mean(dim=0)
        com0x.append(c0[0].item());com1x.append(c1[0].item())
        cdist_h.append(torch.norm(c0-c1).item())
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
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5
        b0fx.append(ext[0,b0m,0].mean().item());b0fy.append(ext[0,b0m,1].mean().item())
        b0fz.append(ext[0,b0m,2].mean().item())
        b1fx.append(ext[0,b1m,0].mean().item());b1fy.append(ext[0,b1m,1].mean().item())
        b1fz.append(ext[0,b1m,2].mean().item())
        if step%20==0: tframes.append(pos[0].cpu().numpy().copy());fframes.append(ext[0].cpu().numpy().copy())
        if cd and len(csa)>n_orig:
            pa=pos[0,csa[n_orig:]];pb=pos[0,csb[n_orig:]]
            spring_h.append((torch.norm(pb-pa,dim=1)-crl[n_orig:]).mean().item())
        else: spring_h.append(0.0)
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
        acc=f*inv_mass.unsqueeze(0).unsqueeze(2)
        vel+=acc*DT;pos+=vel*DT
    
    r_data = {"b0fx":b0fx,"b0fy":b0fy,"b0fz":b0fz,"b1fx":b1fx,"b1fy":b1fy,"b1fz":b1fz,
              "com0x":com0x,"com1x":com1x,"cdist":cdist_h,"spring":spring_h,
              "tframes":tframes,"fframes":fframes,"body_ids":bi}
    corr_x = np.corrcoef(b0fx, b1fx)[0,1]
    corr_y = np.corrcoef(b0fy, b1fy)[0,1]
    r_data["corr_x"] = corr_x; r_data["corr_y"] = corr_y
    return r_data


def main():
    print("="*60)
    print("  ASYMMETRIC BODY FORCE ANALYSIS")
    print("  Does mass asymmetry force role differentiation?")
    print("="*60)
    
    nsteps = 600; gap = 0.5; ngens = 150; psz = 200
    
    conditions = [
        ("Symmetric\n(mass=1.0/1.0)", None, "#27ae60"),
        ("Bone+Muscle\n(mass=2.0/0.5)", [2.0, 0.5], "#e67e22"),
        ("Extreme\n(mass=3.0/0.3)", [3.0, 0.3], "#8e44ad"),
    ]
    
    all_results = {}
    
    for label, masses, color in conditions:
        short = label.split("\n")[0]
        print(f"\n{'='*60}")
        print(f"  Condition: {short} — {masses if masses else 'default'}")
        print(f"{'='*60}")
        
        genes, fit, data = evolve(nsteps, gap, ngens, psz, masses, short)
        torch.cuda.empty_cache()
        
        print(f"  Replaying with force tracking...")
        r = replay_forces(genes, data, nsteps, masses)
        r["fitness"] = fit
        r["masses"] = masses
        all_results[short] = r
        
        print(f"  Force corr X: r={r['corr_x']:.3f}")
        print(f"  Force corr Y: r={r['corr_y']:.3f}")
        print(f"  → {'DIFFERENTIATED!' if abs(r['corr_x']) < 0.3 else 'Synchronized' if r['corr_x'] > 0.5 else 'Weakly correlated'}")
    
    # ============================================
    # MASTER FIGURE: 3-condition comparison
    # ============================================
    fig = plt.figure(figsize=(20, 16))
    
    times = np.arange(nsteps) * DT
    
    for i, (label, masses, color) in enumerate(conditions):
        short = label.split("\n")[0]
        r = all_results[short]
        
        # Row 1: Force X per body
        ax = fig.add_subplot(4, 3, i+1)
        ax.plot(times, r["b0fx"], '#e74c3c', linewidth=1, alpha=0.7, label='Body 0 (heavy)' if masses else 'Body 0')
        ax.plot(times, r["b1fx"], '#3498db', linewidth=1, alpha=0.7, label='Body 1 (light)' if masses else 'Body 1')
        ax.set_ylabel("Force X"); ax.set_title(f"{label}\nFit={r['fitness']:.0f}", fontweight='bold', fontsize=10)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        
        # Row 2: Force Y per body
        ax = fig.add_subplot(4, 3, i+4)
        ax.plot(times, r["b0fy"], '#e74c3c', linewidth=1, alpha=0.7)
        ax.plot(times, r["b1fy"], '#3498db', linewidth=1, alpha=0.7)
        ax.set_ylabel("Force Y"); ax.grid(True, alpha=0.2)
        
        # Row 3: Force X correlation scatter
        ax = fig.add_subplot(4, 3, i+7)
        ax.scatter(r["b0fx"], r["b1fx"], s=1, alpha=0.3, c=times, cmap='viridis')
        lim = max(abs(np.array(r["b0fx"]+r["b1fx"])).max()*1.1, 1)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
        ax.plot([-lim,lim],[-lim,lim],'r--',linewidth=0.5,alpha=0.5)
        ax.set_xlabel("Body 0 Fx"); ax.set_ylabel("Body 1 Fx")
        status = "DIFFERENTIATED!" if abs(r["corr_x"]) < 0.3 else "Synchronized" if r["corr_x"] > 0.5 else "Weakly corr."
        ax.set_title(f"r(Fx) = {r['corr_x']:.3f} [{status}]", fontweight='bold',
                     color='red' if abs(r["corr_x"]) < 0.3 else 'green' if r["corr_x"] > 0.5 else 'orange')
        ax.grid(True, alpha=0.2)
    
    # Row 4: Summary bar chart
    ax = fig.add_subplot(4, 1, 4)
    labels_short = [c[0].split("\n")[0] for c in conditions]
    corrs_x = [all_results[l]["corr_x"] for l in labels_short]
    corrs_y = [all_results[l]["corr_y"] for l in labels_short]
    fits = [all_results[l]["fitness"] for l in labels_short]
    
    x = np.arange(3); w = 0.25
    cols = [c[2] for c in conditions]
    bars1 = ax.bar(x-w, corrs_x, w, label='Corr(Fx)', color=cols, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, corrs_y, w, label='Corr(Fy)', color=cols, alpha=0.4, edgecolor='black')
    for bar, val in zip(bars1, corrs_x):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{val:.3f}",
                ha='center', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, corrs_y):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{val:.3f}",
                ha='center', fontsize=9)
    
    ax2 = ax.twinx()
    ax2.bar(x+w, fits, w, color='gray', alpha=0.3, label='Fitness')
    for xi, val in zip(x+w, fits):
        ax2.text(xi, val+3, f"{val:.0f}", ha='center', fontsize=9, color='gray')
    ax2.set_ylabel("Fitness", color='gray')
    
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n{c[1] if c[1] else '[1.0,1.0]'}" for l,c in zip(labels_short, conditions)])
    ax.set_ylabel("Force Correlation (r)")
    ax.axhline(y=0.5, color='green', linewidth=1, linestyle='--', alpha=0.5, label='Sync threshold')
    ax.axhline(y=0.3, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Diff threshold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylim(-0.5, 1.1)
    ax.legend(loc='lower left', fontsize=8); ax2.legend(loc='lower right', fontsize=8)
    ax.set_title("Role Differentiation Summary: Does Mass Asymmetry → Functional Differentiation?", fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    
    fig.suptitle("Asymmetric Body Force Analysis\nSymmetric vs Bone+Muscle vs Extreme Mass Ratio",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "asymmetric_force_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  Master figure: {path}")
    
    # Force arrow filmstrip for most interesting condition
    # (pick the one with lowest correlation = most differentiation)
    most_diff = min(labels_short, key=lambda l: abs(all_results[l]["corr_x"]))
    r_best = all_results[most_diff]
    nf = len(r_best["tframes"])
    keys = [0, nf//4, nf//2, 3*nf//4, nf-1]
    
    fig2, axes2 = plt.subplots(1, 5, figsize=(25, 5), subplot_kw={'projection': '3d'})
    for ax_i, ki in enumerate(keys):
        ax = axes2[ax_i]
        fp = r_best["tframes"][ki]; ff = r_best["fframes"][ki]; bids = r_best["body_ids"]
        for bi, col in [(0,'#e74c3c'),(1,'#3498db')]:
            m=bids==bi; ax.scatter(fp[m,0],fp[m,2],fp[m,1],c=col,s=8,alpha=0.6)
            cm=fp[m].mean(axis=0);fm=ff[m].mean(axis=0);sc=0.15
            ax.quiver(cm[0],cm[2],cm[1],fm[0]*sc,fm[2]*sc,fm[1]*sc,
                      color=col,arrow_length_ratio=0.3,linewidth=3,alpha=0.9)
        cx=fp[:,0].mean();ax.set_xlim(cx-5,cx+5);ax.set_ylim(-3,3);ax.set_zlim(-2,5)
        ax.set_title(f"t={ki*20*DT:.1f}s",fontsize=10);ax.view_init(elev=20,azim=-60)
    fig2.suptitle(f"Force Arrows: {most_diff} (most differentiated, r={all_results[most_diff]['corr_x']:.3f})",
                  fontsize=14, fontweight='bold')
    path2 = os.path.join(OUTPUT_DIR, "asymmetric_force_arrows.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Arrow filmstrip: {path2}")
    
    # Save log
    log = {
        "experiment": "Asymmetric Body Force Analysis - Role Differentiation Test",
        "conditions": {}
    }
    for label, masses, color in conditions:
        short = label.split("\n")[0]
        r = all_results[short]
        log["conditions"][short] = {
            "masses": masses,
            "fitness": r["fitness"],
            "corr_fx": float(r["corr_x"]),
            "corr_fy": float(r["corr_y"]),
            "differentiated": bool(abs(r["corr_x"]) < 0.3),
        }
    
    # Conclusion
    sym_r = all_results["Symmetric"]["corr_x"]
    bm_r = all_results["Bone+Muscle"]["corr_x"]
    ext_r = all_results["Extreme"]["corr_x"]
    
    if abs(ext_r) < abs(sym_r) - 0.2:
        log["conclusion"] = (
            f"CONFIRMED: Mass asymmetry reduces force correlation! "
            f"Symmetric r={sym_r:.3f} → Extreme r={ext_r:.3f}. "
            f"Asymmetric structure creates evolutionary pressure for role differentiation."
        )
    else:
        log["conclusion"] = (
            f"NEGATIVE: Mass asymmetry did NOT force differentiation. "
            f"Symmetric r={sym_r:.3f}, Extreme r={ext_r:.3f}. "
            f"150 generations may be insufficient, or body_id input doesn't carry enough information."
        )
    
    path_log = os.path.join(RESULTS_DIR, "asymmetric_force_log.json")
    with open(path_log, "w") as f: json.dump(log, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  CONCLUSION")
    print(f"{'='*60}")
    print(f"  {log['conclusion']}")
    print(f"  Log: {path_log}")
    
    # Victory beep
    try:
        import winsound
        for freq in [523,659,784,1047,1319,1568]:
            winsound.Beep(freq, 200); time.sleep(0.05)
        time.sleep(0.3)
        for freq in [1568,1319,1047,784,659,523]:
            winsound.Beep(freq, 200); time.sleep(0.05)
    except: pass
    print("\n  🎉 ASYMMETRIC FORCE ANALYSIS COMPLETE! 🎉")


if __name__ == "__main__":
    main()
