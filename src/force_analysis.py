"""
Force Visualization: NN Role Differentiation Analysis
=======================================================
Evolves a V2.1 combining robot, then replays the best genome
with detailed per-body force tracking to answer:

1. Does the NN use body_id to differentiate forces?
2. Is there a "slingshot" effect (front pulls, back follows)?
3. Are combine springs under tension or compression?
4. Do the two bodies have asymmetric roles?

Generates:
  - Force arrow quiver plots (filmstrip with arrows)
  - Per-body force magnitude over time
  - Spring tension/compression heatmap
  - Role differentiation summary
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import Delaunay
import os, time, json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DT = 0.010; GROUND_Y = -0.5; GROUND_K = 600.0; GRAVITY = -9.8
FRICTION = 3.0; BASE_AMP = 30.0; DRAG = 0.4; SPRING_K = 30.0; SPRING_DAMP = 1.5

INPUT_SIZE = 7; HIDDEN_SIZE = 32; OUTPUT_SIZE = 3
N_W1 = INPUT_SIZE * HIDDEN_SIZE; N_B1 = HIDDEN_SIZE
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE; N_B2 = OUTPUT_SIZE; N_FREQ = 1
N_GENES = N_W1 + N_B1 + N_W2 + N_B2 + N_FREQ


def build_bodies(grid_x, grid_y, grid_z, spacing, gap):
    n_per = grid_x * grid_y * grid_z; n_total = n_per * 2
    all_pos = np.zeros((n_total, 3)); body_ids = np.zeros(n_total, dtype=np.int64)
    body_width = (grid_x - 1) * spacing
    idx = 0
    for bi in range(2):
        for gx in range(grid_x):
            for gy in range(grid_y):
                for gz in range(grid_z):
                    x = (-(gap/2+body_width)+gx*spacing) if bi==0 else ((gap/2+body_width)-gx*spacing)
                    all_pos[idx] = [x, 2.0+gy*spacing, gz*spacing-(grid_z-1)*spacing/2]
                    body_ids[idx] = bi; idx += 1
    sa, sb, rl = [], [], []
    for bi in range(2):
        mask = np.where(body_ids == bi)[0]; bp = all_pos[mask]
        tri = Delaunay(bp); edges = set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1, 4):
                    edges.add((min(mask[s[i]],mask[s[j]]), max(mask[s[i]],mask[s[j]])))
        for a, b in edges:
            sa.append(a); sb.append(b); rl.append(np.linalg.norm(all_pos[a]-all_pos[b]))
    norm_pos = np.zeros_like(all_pos)
    for bi in range(2):
        m = body_ids == bi
        for d in range(3):
            vmin, vmax = all_pos[m,d].min(), all_pos[m,d].max()
            norm_pos[m,d] = 2*(all_pos[m,d]-vmin)/(vmax-vmin+1e-8)-1
    return (all_pos, norm_pos, body_ids, np.array(sa), np.array(sb), np.array(rl), n_per, n_total)


@torch.no_grad()
def simulate_batch(genomes, rp, np_, bi_, sa_, sb_, rl_, N, nper, nsteps):
    B = genomes.shape[0]; pos = rp.unsqueeze(0).expand(B,-1,-1).clone()
    vel = torch.zeros(B,N,3,device=DEVICE)
    idx=0; W1=genomes[:,idx:idx+N_W1].reshape(B,INPUT_SIZE,HIDDEN_SIZE); idx+=N_W1
    b1=genomes[:,idx:idx+N_B1].unsqueeze(1); idx+=N_B1
    W2=genomes[:,idx:idx+N_W2].reshape(B,HIDDEN_SIZE,OUTPUT_SIZE); idx+=N_W2
    b2=genomes[:,idx:idx+N_B2].unsqueeze(1); idx+=N_B2
    freq=genomes[:,idx].abs()
    sx=pos[:,:,0].mean(dim=1); bid=bi_.float().unsqueeze(0).unsqueeze(2).expand(B,N,1)
    ni=np_.unsqueeze(0).expand(B,-1,-1)
    csa=sa_.clone();csb=sb_.clone();crl=rl_.clone()
    comb=torch.zeros(B,1,1,device=DEVICE); te=torch.zeros(B,device=DEVICE)
    b0m=bi_==0;b1m=bi_==1;b0i=b0m.nonzero(as_tuple=True)[0];b1i=b1m.nonzero(as_tuple=True)[0]
    cd=False
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
        ext[:,:,2]=BASE_AMP*o[:,:,2]*gc*0.5; te+=(ext**2).sum(dim=(1,2))
        f=torch.zeros(B,N,3,device=DEVICE);f[:,:,1]+=GRAVITY
        pa=pos[:,csa];pb=pos[:,csb];d=pb-pa;di=torch.norm(d,dim=2,keepdim=True).clamp(min=1e-8)
        dr=d/di;r=crl.unsqueeze(0).unsqueeze(2);s=di-r
        rv=vel[:,csb]-vel[:,csa];va=(rv*dr).sum(dim=2,keepdim=True)
        ft=SPRING_K*s*dr+SPRING_DAMP*va*dr
        f.scatter_add_(1,csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3),ft)
        f.scatter_add_(1,csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3),-ft)
        pen=(GROUND_Y-pos[:,:,1]).clamp(min=0);f[:,:,1]+=GROUND_K*pen
        bl=(pos[:,:,1]<GROUND_Y).float()
        f[:,:,0]-=FRICTION*vel[:,:,0]*bl;f[:,:,2]-=FRICTION*vel[:,:,2]*bl
        f-=DRAG*vel;f+=ext;vel+=f*DT;pos+=vel*DT
    disp=pos[:,:,0].mean(dim=1)-sx;dz=pos[:,:,2].mean(dim=1).abs()
    sp=pos.max(dim=1).values-pos.min(dim=1).values
    spp=((sp-8.0).clamp(min=0)*1.5).sum(dim=1);bw=(pos[:,:,1]<GROUND_Y-1).float().sum(dim=1)*0.2
    me=N*nsteps*(BASE_AMP*1.5)**2*3;ep=1.0*(te/me)*100
    c0=pos[:,b0m].mean(dim=1);c1=pos[:,b1m].mean(dim=1)
    coh=torch.clamp(3.0-torch.norm(c0-c1,dim=1),min=0)*2.0
    return disp-dz-spp-bw-ep+coh, disp


def evolve(nsteps, gap, ngens=150, psz=200):
    gx,gy,gz,sp = 10,5,4,0.35
    data = build_bodies(gx,gy,gz,sp,gap)
    ap,np_,bi,sa,sb,rl,nper,nt = data
    rp=torch.tensor(ap,dtype=torch.float32,device=DEVICE)
    np_t=torch.tensor(np_,dtype=torch.float32,device=DEVICE)
    bi_t=torch.tensor(bi,dtype=torch.long,device=DEVICE)
    sa_t=torch.tensor(sa,dtype=torch.long,device=DEVICE)
    sb_t=torch.tensor(sb,dtype=torch.long,device=DEVICE)
    rl_t=torch.tensor(rl,dtype=torch.float32,device=DEVICE)
    s1=np.sqrt(2.0/(INPUT_SIZE+HIDDEN_SIZE));s2=np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    pop=torch.randn(psz,N_GENES,device=DEVICE)*0.3
    pop[:,:N_W1]*=s1/0.3;pop[:,N_W1:N_W1+N_B1]=0
    pop[:,N_W1+N_B1:N_W1+N_B1+N_W2]*=s2/0.3
    pop[:,N_W1+N_B1+N_W2:N_W1+N_B1+N_W2+N_B2]=0
    pop[:,-1]=torch.empty(psz,device=DEVICE).uniform_(0.5,3.0)
    pf=torch.full((psz,),float('-inf'),device=DEVICE)
    t0=time.time()
    for gen in range(ngens):
        nd=(pf==float('-inf'))
        if nd.any():
            ix=nd.nonzero(as_tuple=True)[0]
            f,d=simulate_batch(pop[ix],rp,np_t,bi_t,sa_t,sb_t,rl_t,nt,nper,nsteps)
            pf[ix]=f
        o=pf.argsort(descending=True);pop=pop[o];pf=pf[o]
        if gen%25==0: print(f"  Gen {gen:3d}/{ngens}: best={pf[0].item():+.2f}")
        ne=max(2,int(psz*0.05));np_=pop[:ne].clone();nf_=pf[:ne].clone()
        nfr=max(2,int(psz*0.05));fr=torch.randn(nfr,N_GENES,device=DEVICE)*0.3
        fr[:,:N_W1]*=s1/0.3;fr[:,-1]=torch.empty(nfr,device=DEVICE).uniform_(0.5,3.0)
        np_=torch.cat([np_,fr]);nf_=torch.cat([nf_,torch.full((nfr,),float('-inf'),device=DEVICE)])
        nc=psz-np_.shape[0]
        t1=torch.randint(psz,(nc,5),device=DEVICE)
        p1=t1[torch.arange(nc,device=DEVICE),pf[t1].argmax(dim=1)]
        t2=torch.randint(psz,(nc,5),device=DEVICE)
        p2=t2[torch.arange(nc,device=DEVICE),pf[t2].argmax(dim=1)]
        mk=torch.rand(nc,N_GENES,device=DEVICE)<0.5
        ch=torch.where(mk,pop[p1],pop[p2])
        mt=torch.rand(nc,N_GENES,device=DEVICE)<0.15
        ch+=torch.randn(nc,N_GENES,device=DEVICE)*0.3*mt.float()
        np_=torch.cat([np_,ch]);nf_=torch.cat([nf_,torch.full((nc,),float('-inf'),device=DEVICE)])
        pop=np_;pf=nf_
    print(f"  Done: {(time.time()-t0)/60:.1f}min | Best={pf[0].item():+.2f}")
    return pop[0].cpu().numpy(), pf[0].item(), data


@torch.no_grad()
def replay_with_forces(best_genes, data, nsteps):
    """Replay with per-particle force tracking."""
    ap, np_a, bi, sa, sb, rl, nper, nt = data
    N = nt
    rp = torch.tensor(ap, dtype=torch.float32, device=DEVICE)
    np_t = torch.tensor(np_a, dtype=torch.float32, device=DEVICE)
    bi_t = torch.tensor(bi, dtype=torch.long, device=DEVICE)
    sa_t = torch.tensor(sa, dtype=torch.long, device=DEVICE)
    sb_t = torch.tensor(sb, dtype=torch.long, device=DEVICE)
    rl_t = torch.tensor(rl, dtype=torch.float32, device=DEVICE)
    
    genes = torch.tensor(best_genes, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    B = 1; pos = rp.unsqueeze(0).clone(); vel = torch.zeros(1, N, 3, device=DEVICE)
    
    idx = 0
    W1 = genes[:, idx:idx+N_W1].reshape(B, INPUT_SIZE, HIDDEN_SIZE); idx += N_W1
    b1 = genes[:, idx:idx+N_B1].unsqueeze(1); idx += N_B1
    W2 = genes[:, idx:idx+N_W2].reshape(B, HIDDEN_SIZE, OUTPUT_SIZE); idx += N_W2
    b2 = genes[:, idx:idx+N_B2].unsqueeze(1); idx += N_B2
    freq = genes[:, idx].abs()
    
    bid = bi_t.float().unsqueeze(0).unsqueeze(2).expand(B, N, 1)
    ni = np_t.unsqueeze(0).expand(B, N, 3)
    b0m = bi_t == 0; b1m = bi_t == 1
    b0i = b0m.nonzero(as_tuple=True)[0]; b1i = b1m.nonzero(as_tuple=True)[0]
    
    csa = sa_t.clone(); csb = sb_t.clone(); crl = rl_t.clone()
    comb = torch.zeros(B, 1, 1, device=DEVICE); cd = False; cstep = None
    n_orig_springs = len(sa)
    
    # Tracking arrays
    body0_force_x, body0_force_y, body0_force_z = [], [], []
    body1_force_x, body1_force_y, body1_force_z = [], [], []
    body0_nn_raw, body1_nn_raw = [], []  # Raw NN output before ground coupling
    com0_x, com1_x, com_dist_hist = [], [], []
    spring_tension_hist = []  # Mean tension of combine springs
    traj_frames, force_frames = [], []
    
    for step in range(nsteps):
        t = step * DT
        
        # Track COMs
        c0 = pos[0, b0m].mean(dim=0); c1 = pos[0, b1m].mean(dim=0)
        com0_x.append(c0[0].item()); com1_x.append(c1[0].item())
        com_dist_hist.append(torch.norm(c0-c1).item())
        
        # Combine
        if not cd and step % 10 == 0:
            p0 = pos[0, b0i]; p1 = pos[0, b1i]
            ds = torch.cdist(p0, p1)
            cl = (ds < 1.2).nonzero(as_tuple=False)
            if cl.shape[0] > 0:
                nn_ = min(cl.shape[0], 500)
                csa = torch.cat([csa, b0i[cl[:nn_, 0]]])
                csb = torch.cat([csb, b1i[cl[:nn_, 1]]])
                crl = torch.cat([crl, ds[cl[:nn_, 0], cl[:nn_, 1]]])
                comb = torch.ones(B, 1, 1, device=DEVICE); cd = True; cstep = step
        
        # NN forward
        st = torch.sin(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        ct = torch.cos(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        nn_in = torch.cat([st, ct, ni, bid, comb.expand(B,N,1)], dim=2)
        h = torch.tanh(torch.bmm(nn_in, W1) + b1)
        o = torch.tanh(torch.bmm(h, W2) + b2)  # (B, N, 3)
        
        # Store raw NN output per body
        body0_nn_raw.append(o[0, b0m].mean(dim=0).cpu().numpy())
        body1_nn_raw.append(o[0, b1m].mean(dim=0).cpu().numpy())
        
        # Apply ground coupling
        og = (pos[:,:,1] < GROUND_Y + 0.3).float()
        gc = 0.5 + og
        ext = torch.zeros(B, N, 3, device=DEVICE)
        ext[:,:,0] = BASE_AMP * o[:,:,0] * gc
        ext[:,:,1] = BASE_AMP * torch.clamp(o[:,:,1], min=0) * gc
        ext[:,:,2] = BASE_AMP * o[:,:,2] * gc * 0.5
        
        # Track per-body mean forces
        body0_force_x.append(ext[0, b0m, 0].mean().item())
        body0_force_y.append(ext[0, b0m, 1].mean().item())
        body0_force_z.append(ext[0, b0m, 2].mean().item())
        body1_force_x.append(ext[0, b1m, 0].mean().item())
        body1_force_y.append(ext[0, b1m, 1].mean().item())
        body1_force_z.append(ext[0, b1m, 2].mean().item())
        
        # Save frames for visualization
        if step % 20 == 0:
            traj_frames.append(pos[0].cpu().numpy().copy())
            force_frames.append(ext[0].cpu().numpy().copy())
        
        # Spring tension tracking (combine springs only)
        if cd and len(csa) > n_orig_springs:
            combine_sa = csa[n_orig_springs:]
            combine_sb = csb[n_orig_springs:]
            combine_rl = crl[n_orig_springs:]
            pa = pos[0, combine_sa]; pb = pos[0, combine_sb]
            actual_dist = torch.norm(pb - pa, dim=1)
            tension = (actual_dist - combine_rl).mean().item()
            spring_tension_hist.append(tension)
        else:
            spring_tension_hist.append(0.0)
        
        # Physics
        f = torch.zeros(B,N,3,device=DEVICE); f[:,:,1] += GRAVITY
        pa = pos[:,csa]; pb = pos[:,csb]; d = pb-pa
        di = torch.norm(d,dim=2,keepdim=True).clamp(min=1e-8)
        dr = d/di; r = crl.unsqueeze(0).unsqueeze(2)
        s = di-r; rv = vel[:,csb]-vel[:,csa]
        va = (rv*dr).sum(dim=2,keepdim=True)
        ft = SPRING_K*s*dr + SPRING_DAMP*va*dr
        f.scatter_add_(1, csa.unsqueeze(0).unsqueeze(2).expand(B,-1,3), ft)
        f.scatter_add_(1, csb.unsqueeze(0).unsqueeze(2).expand(B,-1,3), -ft)
        pen = (GROUND_Y-pos[:,:,1]).clamp(min=0); f[:,:,1] += GROUND_K*pen
        bl = (pos[:,:,1]<GROUND_Y).float()
        f[:,:,0] -= FRICTION*vel[:,:,0]*bl; f[:,:,2] -= FRICTION*vel[:,:,2]*bl
        f -= DRAG*vel; f += ext
        vel += f * DT; pos += vel * DT
    
    return {
        "body0_fx": body0_force_x, "body0_fy": body0_force_y, "body0_fz": body0_force_z,
        "body1_fx": body1_force_x, "body1_fy": body1_force_y, "body1_fz": body1_force_z,
        "body0_nn": np.array(body0_nn_raw), "body1_nn": np.array(body1_nn_raw),
        "com0_x": com0_x, "com1_x": com1_x, "com_dist": com_dist_hist,
        "spring_tension": spring_tension_hist,
        "traj_frames": traj_frames, "force_frames": force_frames,
        "body_ids": bi, "combine_step": cstep,
    }


def main():
    print("="*60)
    print("  FORCE VISUALIZATION: Role Differentiation Analysis")
    print("="*60)
    
    nsteps = 600; gap = 0.5
    
    # Evolve
    print("\n--- Evolving V2.1 (instant combine) ---")
    best_genes, best_fit, data = evolve(nsteps, gap, ngens=150)
    
    torch.cuda.empty_cache()
    
    # Replay with force tracking
    print("\n--- Replaying with force tracking ---")
    r = replay_with_forces(best_genes, data, nsteps)
    cs = r["combine_step"]
    print(f"  Combine at step {cs}")
    
    times = np.arange(nsteps) * DT
    
    # ============================================
    # FIGURE 1: Per-body force over time (6 panels)
    # ============================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    
    # X forces
    ax = axes[0, 0]
    ax.plot(times, r["body0_fx"], '#e74c3c', linewidth=1.5, label='Body 0 (left)', alpha=0.8)
    ax.plot(times, r["body1_fx"], '#3498db', linewidth=1.5, label='Body 1 (right)', alpha=0.8)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_ylabel("Force X"); ax.set_title("NN External Force (X direction)", fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
    
    # X force difference
    ax = axes[0, 1]
    diff_x = np.array(r["body0_fx"]) - np.array(r["body1_fx"])
    ax.fill_between(times, 0, diff_x, where=diff_x>0, color='#e74c3c', alpha=0.3, label='Body0 > Body1')
    ax.fill_between(times, 0, diff_x, where=diff_x<0, color='#3498db', alpha=0.3, label='Body1 > Body0')
    ax.plot(times, diff_x, 'black', linewidth=1)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_ylabel("Fx Difference"); ax.set_title("Force X Asymmetry (Body0 - Body1)", fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    
    # Y forces
    ax = axes[1, 0]
    ax.plot(times, r["body0_fy"], '#e74c3c', linewidth=1.5, label='Body 0', alpha=0.8)
    ax.plot(times, r["body1_fy"], '#3498db', linewidth=1.5, label='Body 1', alpha=0.8)
    ax.set_ylabel("Force Y"); ax.set_title("NN External Force (Y / Up)", fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
    
    # Spring tension
    ax = axes[1, 1]
    ax.plot(times, r["spring_tension"], '#2ecc71', linewidth=2)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.fill_between(times, 0, r["spring_tension"],
                     where=np.array(r["spring_tension"])>0,
                     color='#e74c3c', alpha=0.2, label='Tension (pulling apart)')
    ax.fill_between(times, 0, r["spring_tension"],
                     where=np.array(r["spring_tension"])<0,
                     color='#3498db', alpha=0.2, label='Compression (pushing together)')
    ax.set_ylabel("Mean Spring Stretch"); ax.set_title("Combine Spring Tension/Compression", fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    
    # Raw NN output comparison
    ax = axes[2, 0]
    nn0 = r["body0_nn"]; nn1 = r["body1_nn"]
    ax.plot(times, nn0[:, 0], '#e74c3c', linewidth=1, alpha=0.7, label='B0 out[0]')
    ax.plot(times, nn1[:, 0], '#3498db', linewidth=1, alpha=0.7, label='B1 out[0]')
    ax.plot(times, nn0[:, 1], '#e74c3c', linewidth=1, alpha=0.4, linestyle='--', label='B0 out[1]')
    ax.plot(times, nn1[:, 1], '#3498db', linewidth=1, alpha=0.4, linestyle='--', label='B1 out[1]')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("NN Output (tanh)")
    ax.set_title("Raw NN Output: Does body_id matter?", fontweight='bold')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)
    
    # COM and distance
    ax = axes[2, 1]
    ax.plot(times, r["com0_x"], '#e74c3c', linewidth=1.5, label='Body 0 COM', alpha=0.7)
    ax.plot(times, r["com1_x"], '#3498db', linewidth=1.5, label='Body 1 COM', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(times, r["com_dist"], '#2ecc71', linewidth=2, label='Distance', alpha=0.5)
    ax2.set_ylabel("Inter-body distance", color='#2ecc71')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("X Position")
    ax.set_title("COM Positions & Inter-body Distance", fontweight='bold')
    ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)
    
    fig.suptitle(f"Force Analysis: V2.1 Combining Robot | Best={best_fit:.1f}\n"
                 f"400 particles, combine@step={cs}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "force_analysis.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  Figure 1: {path1}")
    
    # ============================================
    # FIGURE 2: Force arrow filmstrip
    # ============================================
    n_frames = len(r["traj_frames"])
    key_idx = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), subplot_kw={'projection': '3d'})
    for ax_i, ki in enumerate(key_idx):
        ax = axes[ax_i]
        fp = r["traj_frames"][ki]
        ff = r["force_frames"][ki]
        bids = r["body_ids"]
        
        for bi, col in [(0, '#e74c3c'), (1, '#3498db')]:
            m = bids == bi
            ax.scatter(fp[m,0], fp[m,2], fp[m,1], c=col, s=8, alpha=0.6)
            # Average force arrow per body
            cm = fp[m].mean(axis=0)
            fm = ff[m].mean(axis=0)
            scale = 0.15
            ax.quiver(cm[0], cm[2], cm[1],
                      fm[0]*scale, fm[2]*scale, fm[1]*scale,
                      color=col, arrow_length_ratio=0.3, linewidth=3, alpha=0.9)
        
        t_sec = ki * 20 * DT
        ax.set_title(f"t={t_sec:.1f}s", fontsize=10)
        cx = fp[:,0].mean()
        ax.set_xlim(cx-5, cx+5); ax.set_ylim(-3, 3); ax.set_zlim(-2, 5)
        ax.view_init(elev=20, azim=-60)
    
    fig.suptitle(f"Force Arrow Filmstrip | Arrows = Mean NN Force per Body",
                 fontsize=14, fontweight='bold')
    path2 = os.path.join(OUTPUT_DIR, "force_arrows.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Figure 2: {path2}")
    
    # ============================================
    # FIGURE 3: Role differentiation summary
    # ============================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Mean force magnitude per body over time (smoothed)
    ax = axes[0]
    window = 30
    b0_mag = np.sqrt(np.array(r["body0_fx"])**2 + np.array(r["body0_fy"])**2 + np.array(r["body0_fz"])**2)
    b1_mag = np.sqrt(np.array(r["body1_fx"])**2 + np.array(r["body1_fy"])**2 + np.array(r["body1_fz"])**2)
    b0_smooth = np.convolve(b0_mag, np.ones(window)/window, mode='valid')
    b1_smooth = np.convolve(b1_mag, np.ones(window)/window, mode='valid')
    t_smooth = times[:len(b0_smooth)]
    ax.plot(t_smooth, b0_smooth, '#e74c3c', linewidth=2, label='Body 0')
    ax.plot(t_smooth, b1_smooth, '#3498db', linewidth=2, label='Body 1')
    ax.fill_between(t_smooth, b0_smooth, b1_smooth, alpha=0.15, color='gray')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("|Force| (smoothed)")
    ax.set_title("Force Magnitude: Who pushes harder?", fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
    
    # Panel B: Force direction polar histogram
    ax = axes[1]
    # Compute force direction (X vs Y) for each body
    fx0 = np.array(r["body0_fx"]); fy0 = np.array(r["body0_fy"])
    fx1 = np.array(r["body1_fx"]); fy1 = np.array(r["body1_fy"])
    angle0 = np.arctan2(fy0, fx0); angle1 = np.arctan2(fy1, fx1)
    bins = np.linspace(-np.pi, np.pi, 37)
    ax.hist(angle0, bins=bins, alpha=0.5, color='#e74c3c', label='Body 0', density=True)
    ax.hist(angle1, bins=bins, alpha=0.5, color='#3498db', label='Body 1', density=True)
    ax.axvline(x=0, color='black', linewidth=0.5, linestyle='--', label='Pure +X')
    ax.axvline(x=np.pi/2, color='gray', linewidth=0.5, linestyle='--', label='Pure +Y')
    ax.set_xlabel("Force Direction (rad)"); ax.set_ylabel("Density")
    ax.set_title("Force Direction Distribution (XY plane)", fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    
    # Panel C: Correlation between bodies
    ax = axes[2]
    ax.scatter(fx0, fx1, s=1, alpha=0.3, c=times, cmap='viridis')
    ax.set_xlabel("Body 0 Force X"); ax.set_ylabel("Body 1 Force X")
    corr = np.corrcoef(fx0, fx1)[0, 1]
    ax.set_title(f"Force X Correlation: r = {corr:.3f}\n"
                 f"{'Synchronized' if corr > 0.5 else 'Independent' if abs(corr) < 0.3 else 'Anti-correlated'}",
                 fontweight='bold')
    lim = max(abs(np.array(list(fx0)+list(fx1))).max() * 1.1, 1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.2)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, times[-1]))
    sm.set_array([]); plt.colorbar(sm, ax=ax, label='Time (s)')
    
    fig.suptitle("Role Differentiation Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "role_differentiation.png")
    plt.savefig(path3, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Figure 3: {path3}")
    
    # Save log
    log = {
        "experiment": "Force Visualization & Role Differentiation",
        "best_fitness": float(best_fit),
        "combine_step": cs,
        "force_correlation_x": float(corr),
        "mean_force_body0": float(b0_mag.mean()),
        "mean_force_body1": float(b1_mag.mean()),
        "mean_nn_diff_x": float(np.abs(np.array(r["body0_fx"]) - np.array(r["body1_fx"])).mean()),
        "spring_tension_mean": float(np.mean(r["spring_tension"])),
        "role_analysis": (
            "SYNCHRONIZED" if corr > 0.5 else
            "DIFFERENTIATED" if abs(corr) < 0.3 else
            "WEAKLY_CORRELATED"
        ),
    }
    log_path = os.path.join(RESULTS_DIR, "force_analysis_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Log: {log_path}")
    
    # Conclusion
    print(f"\n{'='*60}")
    print(f"  ROLE DIFFERENTIATION RESULTS")
    print(f"{'='*60}")
    print(f"  Force correlation (X): r = {corr:.3f}")
    print(f"  Mean |F| Body 0: {b0_mag.mean():.2f}")
    print(f"  Mean |F| Body 1: {b1_mag.mean():.2f}")
    print(f"  Mean spring tension: {np.mean(r['spring_tension']):.4f}")
    if corr > 0.5:
        print(f"  → Bodies are SYNCHRONIZED: same phase, same direction")
        print(f"    The NN treats them as one big body (body_id not used)")
    elif abs(corr) < 0.3:
        print(f"  → Bodies have DIFFERENTIATED roles!")
        print(f"    The NN uses body_id to assign different functions")
    else:
        print(f"  → Bodies are WEAKLY CORRELATED")
    
    try:
        import winsound
        for f in [523, 659, 784, 1047]: winsound.Beep(f, 300); time.sleep(0.1)
    except: pass
    print("\n  FORCE ANALYSIS COMPLETE!")


if __name__ == "__main__":
    main()
