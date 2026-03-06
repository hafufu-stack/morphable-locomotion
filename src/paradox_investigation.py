"""
Dual-Mode Reward Paradox Investigation
========================================
Investigates why gap=5.0 achieved highest fitness (+346) despite gap=1.0
having highest displacement (290).

Analysis 1: Re-evolve with decomposed fitness tracking
  - Track approach_reward, combine_bonus, displacement, penalties separately
  - Compare "pure locomotion score" across 3 gap conditions

Analysis 2: Trajectory visualization for all 3 gap conditions
  - Frame-by-frame replay showing approach→combine→walk phases
  - COM distance plot with phase annotations
  - Body separation heatmap over time

Autonomous: runs to completion, beeps when done.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import os, time, json, glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANIM_DIR = os.path.join(OUTPUT_DIR, "paradox_anim")
os.makedirs(ANIM_DIR, exist_ok=True)

# Physics
DT = 0.010; GROUND_Y = -0.5; GROUND_K = 600.0; GRAVITY = -9.8
FRICTION = 3.0; BASE_AMP = 30.0; DRAG = 0.4; SPRING_K = 30.0; SPRING_DAMP = 1.5

# NN
INPUT_SIZE = 7; HIDDEN_SIZE = 32; OUTPUT_SIZE = 3
N_W1 = INPUT_SIZE * HIDDEN_SIZE; N_B1 = HIDDEN_SIZE
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE; N_B2 = OUTPUT_SIZE; N_FREQ = 1
N_GENES = N_W1 + N_B1 + N_W2 + N_B2 + N_FREQ


def beep():
    try:
        import winsound
        for f in [523, 659, 784, 1047]: winsound.Beep(f, 300); time.sleep(0.1)
    except: pass


def build_bodies(grid_x, grid_y, grid_z, spacing, gap):
    n_per = grid_x * grid_y * grid_z
    n_total = n_per * 2
    all_pos = np.zeros((n_total, 3))
    body_ids = np.zeros(n_total, dtype=np.int64)
    body_width = (grid_x - 1) * spacing
    
    idx = 0
    for bi in range(2):
        for gx in range(grid_x):
            for gy in range(grid_y):
                for gz in range(grid_z):
                    if bi == 0:
                        x = -(gap/2 + body_width) + gx * spacing
                    else:
                        x = (gap/2 + body_width) - gx * spacing
                    all_pos[idx] = [x, 2.0 + gy * spacing,
                                     gz * spacing - (grid_z-1)*spacing/2]
                    body_ids[idx] = bi
                    idx += 1
    
    springs_a, springs_b, rest_lengths = [], [], []
    for bi in range(2):
        mask = np.where(body_ids == bi)[0]
        bp = all_pos[mask]
        tri = Delaunay(bp)
        edges = set()
        for s in tri.simplices:
            for i in range(4):
                for j in range(i+1, 4):
                    edges.add((min(mask[s[i]], mask[s[j]]),
                               max(mask[s[i]], mask[s[j]])))
        for a, b in edges:
            springs_a.append(a); springs_b.append(b)
            rest_lengths.append(np.linalg.norm(all_pos[a]-all_pos[b]))
    
    norm_pos = np.zeros_like(all_pos)
    for bi in range(2):
        m = body_ids == bi
        for d in range(3):
            vmin, vmax = all_pos[m, d].min(), all_pos[m, d].max()
            norm_pos[m, d] = 2*(all_pos[m, d]-vmin)/(vmax-vmin+1e-8)-1
    
    return (all_pos, norm_pos, body_ids,
            np.array(springs_a), np.array(springs_b), np.array(rest_lengths),
            n_per, n_total)


@torch.no_grad()
def simulate_decomposed(genomes, rest_pos_t, norm_pos_t, body_ids_t,
                        spring_a_t, spring_b_t, rest_lengths_t,
                        n_particles, n_per_body, n_steps,
                        combine_dist=1.2, max_new_springs=500):
    """Simulate with DECOMPOSED fitness tracking."""
    B = genomes.shape[0]; N = n_particles
    pos = rest_pos_t.unsqueeze(0).expand(B, -1, -1).clone()
    vel = torch.zeros(B, N, 3, device=DEVICE)
    
    idx = 0
    W1 = genomes[:, idx:idx+N_W1].reshape(B, INPUT_SIZE, HIDDEN_SIZE); idx += N_W1
    b1 = genomes[:, idx:idx+N_B1].unsqueeze(1); idx += N_B1
    W2 = genomes[:, idx:idx+N_W2].reshape(B, HIDDEN_SIZE, OUTPUT_SIZE); idx += N_W2
    b2 = genomes[:, idx:idx+N_B2].unsqueeze(1); idx += N_B2
    freq = genomes[:, idx].abs()
    
    start_com_x = pos[:, :, 0].mean(dim=1)
    body_id_input = body_ids_t.float().unsqueeze(0).unsqueeze(2).expand(B, N, 1)
    norm_inp = norm_pos_t.unsqueeze(0).expand(B, -1, -1)
    
    cur_sa = spring_a_t.clone(); cur_sb = spring_b_t.clone()
    cur_rl = rest_lengths_t.clone()
    combined = torch.zeros(B, 1, 1, device=DEVICE)
    total_energy = torch.zeros(B, device=DEVICE)
    
    body0_mask = (body_ids_t == 0); body1_mask = (body_ids_t == 1)
    body0_idx = body0_mask.nonzero(as_tuple=True)[0]
    body1_idx = body1_mask.nonzero(as_tuple=True)[0]
    combine_done = False
    
    # Decomposed tracking
    approach_reward = torch.zeros(B, device=DEVICE)
    combine_bonus_val = torch.zeros(B, device=DEVICE)
    
    for step in range(n_steps):
        t = step * DT
        if not combine_done and step % 10 == 0:
            p0 = pos[0, body0_idx]; p1 = pos[0, body1_idx]
            dists = torch.cdist(p0, p1)
            close = (dists < combine_dist).nonzero(as_tuple=False)
            if close.shape[0] > 0:
                n_new = min(close.shape[0], max_new_springs)
                na = body0_idx[close[:n_new, 0]]; nb = body1_idx[close[:n_new, 1]]
                nr = dists[close[:n_new, 0], close[:n_new, 1]]
                cur_sa = torch.cat([cur_sa, na]); cur_sb = torch.cat([cur_sb, nb])
                cur_rl = torch.cat([cur_rl, nr])
                combined = torch.ones(B, 1, 1, device=DEVICE)
                combine_done = True
                combine_bonus_val = torch.full((B,), 30.0, device=DEVICE)
        
        if not combine_done and step % 20 == 0:
            c0 = pos[:, body0_mask].mean(dim=1)
            c1 = pos[:, body1_mask].mean(dim=1)
            approach_reward += torch.clamp(5.0 - torch.norm(c0-c1, dim=1), min=0)
        
        sin_t = torch.sin(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        cos_t = torch.cos(2*np.pi*freq*t).reshape(B,1,1).expand(B,N,1)
        nn_in = torch.cat([sin_t, cos_t, norm_inp, body_id_input,
                           combined.expand(B,N,1)], dim=2)
        hidden = torch.tanh(torch.bmm(nn_in, W1) + b1)
        output = torch.tanh(torch.bmm(hidden, W2) + b2)
        on_ground = (pos[:,:,1] < GROUND_Y + 0.3).float()
        gc = 0.5 + 1.0 * on_ground
        ext = torch.zeros(B, N, 3, device=DEVICE)
        ext[:,:,0] = BASE_AMP * output[:,:,0] * gc
        ext[:,:,1] = BASE_AMP * torch.clamp(output[:,:,1], min=0) * gc
        ext[:,:,2] = BASE_AMP * output[:,:,2] * gc * 0.5
        total_energy += (ext**2).sum(dim=(1,2))
        
        forces = torch.zeros(B, N, 3, device=DEVICE); forces[:,:,1] += GRAVITY
        pa = pos[:, cur_sa]; pb = pos[:, cur_sb]
        diff = pb - pa; dist = torch.norm(diff, dim=2, keepdim=True).clamp(min=1e-8)
        direction = diff / dist; rest = cur_rl.unsqueeze(0).unsqueeze(2)
        stretch = dist - rest; rv = vel[:, cur_sb] - vel[:, cur_sa]
        va = (rv * direction).sum(dim=2, keepdim=True)
        ft = SPRING_K*stretch*direction + SPRING_DAMP*va*direction
        forces.scatter_add_(1, cur_sa.unsqueeze(0).unsqueeze(2).expand(B,-1,3), ft)
        forces.scatter_add_(1, cur_sb.unsqueeze(0).unsqueeze(2).expand(B,-1,3), -ft)
        pen = (GROUND_Y - pos[:,:,1]).clamp(min=0); forces[:,:,1] += GROUND_K*pen
        bl = (pos[:,:,1] < GROUND_Y).float()
        forces[:,:,0] -= FRICTION*vel[:,:,0]*bl; forces[:,:,2] -= FRICTION*vel[:,:,2]*bl
        forces -= DRAG*vel; forces += ext
        vel += forces * DT; pos += vel * DT
    
    # Decomposed fitness
    displacement = pos[:,:,0].mean(dim=1) - start_com_x
    drift_z = pos[:,:,2].mean(dim=1).abs()
    spread = pos.max(dim=1).values - pos.min(dim=1).values
    spread_pen = ((spread - 8.0).clamp(min=0)*1.5).sum(dim=1)
    below = (pos[:,:,1] < GROUND_Y-1.0).float().sum(dim=1)*0.2
    max_e = N * n_steps * (BASE_AMP*1.5)**2 * 3
    energy_pen = 1.0 * (total_energy/max_e) * 100
    com0 = pos[:, body0_mask].mean(dim=1); com1 = pos[:, body1_mask].mean(dim=1)
    merge_dist = torch.norm(com0-com1, dim=1)
    cohesion = torch.clamp(3.0 - merge_dist, min=0) * 2.0
    
    penalties = drift_z + spread_pen + below + energy_pen
    
    # Total fitness (same as dual_mode)
    fitness = displacement - penalties + cohesion + approach_reward + combine_bonus_val
    
    # Pure locomotion (no approach/combine bonuses)
    pure_locomotion = displacement - penalties + cohesion
    
    return {
        "fitness": fitness,
        "pure_locomotion": pure_locomotion,
        "displacement": displacement,
        "approach_reward": approach_reward,
        "combine_bonus": combine_bonus_val,
        "penalties": penalties,
        "cohesion": cohesion,
    }


def evolve_decomposed(n_steps, gap, n_gens=200, pop_size=200, label=""):
    """Evolve with decomposed fitness tracking."""
    grid_x, grid_y, grid_z, spacing = 10, 5, 4, 0.35
    data = build_bodies(grid_x, grid_y, grid_z, spacing, gap)
    all_pos, norm_pos, body_ids, sa, sb, rl, n_per, n_total = data
    
    rest_pos_t = torch.tensor(all_pos, dtype=torch.float32, device=DEVICE)
    norm_pos_t = torch.tensor(norm_pos, dtype=torch.float32, device=DEVICE)
    body_ids_t = torch.tensor(body_ids, dtype=torch.long, device=DEVICE)
    spring_a_t = torch.tensor(sa, dtype=torch.long, device=DEVICE)
    spring_b_t = torch.tensor(sb, dtype=torch.long, device=DEVICE)
    rest_len_t = torch.tensor(rl, dtype=torch.float32, device=DEVICE)
    
    s1 = np.sqrt(2.0/(INPUT_SIZE+HIDDEN_SIZE))
    s2 = np.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE))
    pop = torch.randn(pop_size, N_GENES, device=DEVICE) * 0.3
    pop[:, :N_W1] *= s1/0.3
    pop[:, N_W1:N_W1+N_B1] = 0
    pop[:, N_W1+N_B1:N_W1+N_B1+N_W2] *= s2/0.3
    pop[:, N_W1+N_B1+N_W2:N_W1+N_B1+N_W2+N_B2] = 0
    pop[:, -1] = torch.empty(pop_size, device=DEVICE).uniform_(0.5, 3.0)
    
    pop_fit = torch.full((pop_size,), float('-inf'), device=DEVICE)
    history = {"best": [], "avg": []}
    t0 = time.time()
    
    for gen in range(n_gens):
        need = (pop_fit == float('-inf'))
        if need.any():
            idxn = need.nonzero(as_tuple=True)[0]
            results = simulate_decomposed(pop[idxn], rest_pos_t, norm_pos_t, body_ids_t,
                                          spring_a_t, spring_b_t, rest_len_t,
                                          n_total, n_per, n_steps)
            pop_fit[idxn] = results["fitness"]
        
        order = pop_fit.argsort(descending=True)
        pop = pop[order]; pop_fit = pop_fit[order]
        history["best"].append(pop_fit[0].item())
        history["avg"].append(pop_fit.mean().item())
        
        if gen % 20 == 0:
            print(f"  Gen {gen:3d}/{n_gens}: best={pop_fit[0].item():+8.2f}  [{(time.time()-t0)/(gen+1):.1f}s/gen]")
        
        ne = max(2, int(pop_size * 0.05))
        new_pop = pop[:ne].clone(); new_fit = pop_fit[:ne].clone()
        nf = max(2, int(pop_size * 0.05))
        fresh = torch.randn(nf, N_GENES, device=DEVICE) * 0.3
        fresh[:, :N_W1] *= s1/0.3; fresh[:, -1] = torch.empty(nf, device=DEVICE).uniform_(0.5, 3.0)
        new_pop = torch.cat([new_pop, fresh])
        new_fit = torch.cat([new_fit, torch.full((nf,), float('-inf'), device=DEVICE)])
        nc = pop_size - new_pop.shape[0]
        t1 = torch.randint(pop_size, (nc, 5), device=DEVICE)
        p1_idx = t1[torch.arange(nc, device=DEVICE), pop_fit[t1].argmax(dim=1)]
        t2 = torch.randint(pop_size, (nc, 5), device=DEVICE)
        p2_idx = t2[torch.arange(nc, device=DEVICE), pop_fit[t2].argmax(dim=1)]
        mask = torch.rand(nc, N_GENES, device=DEVICE) < 0.5
        children = torch.where(mask, pop[p1_idx], pop[p2_idx])
        mut = torch.rand(nc, N_GENES, device=DEVICE) < 0.15
        children += torch.randn(nc, N_GENES, device=DEVICE) * 0.3 * mut.float()
        new_pop = torch.cat([new_pop, children])
        new_fit = torch.cat([new_fit, torch.full((nc,), float('-inf'), device=DEVICE)])
        pop = new_pop; pop_fit = new_fit
    
    # Final decomposed evaluation of best
    best_results = simulate_decomposed(pop[:1], rest_pos_t, norm_pos_t, body_ids_t,
                                        spring_a_t, spring_b_t, rest_len_t,
                                        n_total, n_per, n_steps)
    
    total = time.time() - t0
    print(f"  Done: {total/60:.1f}min | Fitness={pop_fit[0].item():+.2f}")
    
    decomp = {k: v[0].item() for k, v in best_results.items()}
    return pop[0].cpu().numpy(), decomp, history, data


@torch.no_grad()
def replay_with_tracking(best_genes, data, n_steps, combine_dist=1.2, max_new_springs=500):
    """Replay best genome with detailed per-step tracking."""
    all_pos, norm_pos, body_ids, sa, sb, rl, n_per, n_total = data
    N = n_total
    
    rest_pos_t = torch.tensor(all_pos, dtype=torch.float32, device=DEVICE)
    norm_pos_t = torch.tensor(norm_pos, dtype=torch.float32, device=DEVICE)
    body_ids_t = torch.tensor(body_ids, dtype=torch.long, device=DEVICE)
    spring_a_t = torch.tensor(sa, dtype=torch.long, device=DEVICE)
    spring_b_t = torch.tensor(sb, dtype=torch.long, device=DEVICE)
    rest_len_t = torch.tensor(rl, dtype=torch.float32, device=DEVICE)
    
    genes_t = torch.tensor(best_genes, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    B = 1; pos = rest_pos_t.unsqueeze(0).clone()
    vel = torch.zeros(1, N, 3, device=DEVICE)
    
    idx = 0
    W1 = genes_t[:, idx:idx+N_W1].reshape(B, INPUT_SIZE, HIDDEN_SIZE); idx += N_W1
    b1g = genes_t[:, idx:idx+N_B1].unsqueeze(1); idx += N_B1
    W2 = genes_t[:, idx:idx+N_W2].reshape(B, HIDDEN_SIZE, OUTPUT_SIZE); idx += N_W2
    b2g = genes_t[:, idx:idx+N_B2].unsqueeze(1); idx += N_B2
    freq_v = genes_t[:, idx].abs()
    
    body_id_inp = body_ids_t.float().unsqueeze(0).unsqueeze(2).expand(B, N, 1)
    norm_inp = norm_pos_t.unsqueeze(0).expand(B, N, 3)
    body0_m = body_ids_t == 0; body1_m = body_ids_t == 1
    body0_idx = body0_m.nonzero(as_tuple=True)[0]
    body1_idx = body1_m.nonzero(as_tuple=True)[0]
    
    cur_sa = spring_a_t.clone(); cur_sb = spring_b_t.clone(); cur_rl = rest_len_t.clone()
    combined_flag = torch.zeros(B, 1, 1, device=DEVICE)
    combine_done = False; combine_step = None
    
    # Per-step tracking
    com0_x_hist, com1_x_hist, com_dist_hist = [], [], []
    total_com_x_hist = []
    traj_frames = []
    
    for step in range(n_steps):
        t = step * DT
        
        # Track COMs
        c0 = pos[0, body0_m].mean(dim=0)
        c1 = pos[0, body1_m].mean(dim=0)
        com0_x_hist.append(c0[0].item())
        com1_x_hist.append(c1[0].item())
        com_dist_hist.append(torch.norm(c0-c1).item())
        total_com_x_hist.append(pos[0, :, 0].mean().item())
        
        if step % 10 == 0:
            traj_frames.append(pos[0].cpu().numpy().copy())
        
        if not combine_done and step % 10 == 0:
            p0 = pos[0, body0_idx]; p1 = pos[0, body1_idx]
            ds = torch.cdist(p0, p1)
            cl = (ds < combine_dist).nonzero(as_tuple=False)
            if cl.shape[0] > 0:
                nn = min(cl.shape[0], max_new_springs)
                na = body0_idx[cl[:nn, 0]]; nb = body1_idx[cl[:nn, 1]]
                nr = ds[cl[:nn, 0], cl[:nn, 1]]
                cur_sa = torch.cat([cur_sa, na]); cur_sb = torch.cat([cur_sb, nb])
                cur_rl = torch.cat([cur_rl, nr])
                combined_flag = torch.ones(B, 1, 1, device=DEVICE)
                combine_done = True; combine_step = step
        
        sin_t = torch.sin(2*np.pi*freq_v*t).reshape(B,1,1).expand(B,N,1)
        cos_t = torch.cos(2*np.pi*freq_v*t).reshape(B,1,1).expand(B,N,1)
        nn_in = torch.cat([sin_t, cos_t, norm_inp, body_id_inp,
                           combined_flag.expand(B,N,1)], dim=2)
        hid = torch.tanh(torch.bmm(nn_in, W1) + b1g)
        out = torch.tanh(torch.bmm(hid, W2) + b2g)
        og = (pos[:,:,1] < GROUND_Y+0.3).float(); gc = 0.5 + og
        ext = torch.zeros(B,N,3, device=DEVICE)
        ext[:,:,0] = BASE_AMP * out[:,:,0] * gc
        ext[:,:,1] = BASE_AMP * torch.clamp(out[:,:,1], min=0) * gc
        ext[:,:,2] = BASE_AMP * out[:,:,2] * gc * 0.5
        
        forces = torch.zeros(B,N,3, device=DEVICE); forces[:,:,1] += GRAVITY
        pa = pos[:,cur_sa]; pb = pos[:,cur_sb]
        diff = pb-pa; dist = torch.norm(diff,dim=2,keepdim=True).clamp(min=1e-8)
        direction = diff/dist; rest = cur_rl.unsqueeze(0).unsqueeze(2)
        stretch = dist-rest; rv = vel[:,cur_sb]-vel[:,cur_sa]
        va_s = (rv*direction).sum(dim=2,keepdim=True)
        ft = SPRING_K*stretch*direction + SPRING_DAMP*va_s*direction
        forces.scatter_add_(1, cur_sa.unsqueeze(0).unsqueeze(2).expand(B,-1,3), ft)
        forces.scatter_add_(1, cur_sb.unsqueeze(0).unsqueeze(2).expand(B,-1,3), -ft)
        pen = (GROUND_Y-pos[:,:,1]).clamp(min=0); forces[:,:,1] += GROUND_K*pen
        bl = (pos[:,:,1] < GROUND_Y).float()
        forces[:,:,0] -= FRICTION*vel[:,:,0]*bl; forces[:,:,2] -= FRICTION*vel[:,:,2]*bl
        forces -= DRAG*vel; forces += ext
        vel += forces * DT; pos += vel * DT
    
    return {
        "com0_x": com0_x_hist, "com1_x": com1_x_hist,
        "com_dist": com_dist_hist, "total_com_x": total_com_x_hist,
        "traj_frames": traj_frames, "body_ids": body_ids,
        "combine_step": combine_step,
    }


def main():
    print("="*60)
    print("  DUAL-MODE REWARD PARADOX INVESTIGATION")
    print("="*60)
    
    n_steps = 800
    gaps = [1.0, 3.0, 5.0]
    labels = ["Close (1.0)", "Medium (3.0)", "Far (5.0)"]
    colors = ['#27ae60', '#e67e22', '#8e44ad']
    
    all_decomp = {}
    all_genes = {}
    all_data = {}
    all_hist = {}
    
    # === ANALYSIS 1: Evolve with decomposed tracking ===
    print("\n" + "="*60)
    print("  ANALYSIS 1: Decomposed Fitness Tracking")
    print("="*60)
    
    for gap, label in zip(gaps, labels):
        print(f"\n--- Gap={gap} ({label}) ---")
        genes, decomp, history, data = evolve_decomposed(n_steps, gap, n_gens=200, label=label)
        all_decomp[gap] = decomp
        all_genes[gap] = genes
        all_data[gap] = data
        all_hist[gap] = history
        
        print(f"  DECOMPOSITION:")
        print(f"    Fitness (total):    {decomp['fitness']:+.2f}")
        print(f"    Pure locomotion:    {decomp['pure_locomotion']:+.2f}")
        print(f"    Displacement:       {decomp['displacement']:+.2f}")
        print(f"    Approach reward:    {decomp['approach_reward']:+.2f}")
        print(f"    Combine bonus:      {decomp['combine_bonus']:+.2f}")
        print(f"    Penalties:          {decomp['penalties']:+.2f}")
        print(f"    Cohesion:           {decomp['cohesion']:+.2f}")
    
    torch.cuda.empty_cache()
    
    # === ANALYSIS 2: Trajectory visualization ===
    print("\n" + "="*60)
    print("  ANALYSIS 2: Trajectory Visualization")
    print("="*60)
    
    all_tracking = {}
    for gap, label in zip(gaps, labels):
        print(f"\n--- Replaying Gap={gap} ({label}) ---")
        tracking = replay_with_tracking(all_genes[gap], all_data[gap], n_steps)
        all_tracking[gap] = tracking
        cs = tracking["combine_step"]
        print(f"  Combine at step {cs} ({cs*DT:.2f}s)" if cs else "  No combine")
    
    # === FIGURE 1: Decomposed fitness bar chart ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel A: Total vs Pure Locomotion comparison
    ax = axes[0]
    x = np.arange(3)
    total_vals = [all_decomp[g]["fitness"] for g in gaps]
    pure_vals = [all_decomp[g]["pure_locomotion"] for g in gaps]
    w = 0.35
    bars1 = ax.bar(x - w/2, total_vals, w, label='Total Fitness', color=colors, alpha=0.8)
    bars2 = ax.bar(x + w/2, pure_vals, w, label='Pure Locomotion', color=colors, alpha=0.4,
                   edgecolor='black', linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels([f"gap={g}" for g in gaps])
    ax.set_ylabel("Score"); ax.set_title("Total Fitness vs Pure Locomotion\n(approach_reward removed)", fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2, axis='y')
    # Annotate with values
    for bar, val in zip(bars1, total_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+3, f"{val:.0f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, pure_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+3, f"{val:.0f}",
                ha='center', va='bottom', fontsize=9)
    
    # Panel B: Stacked component breakdown
    ax = axes[1]
    components = ['displacement', 'approach_reward', 'combine_bonus', 'cohesion']
    comp_colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    bottoms = np.zeros(3)
    for comp, cc in zip(components, comp_colors):
        vals = [max(0, all_decomp[g][comp]) for g in gaps]
        ax.bar(x, vals, 0.5, bottom=bottoms, label=comp.replace('_', ' ').title(), color=cc, alpha=0.8)
        bottoms += vals
    pen_vals = [all_decomp[g]["penalties"] for g in gaps]
    ax.bar(x, [-v for v in pen_vals], 0.5, label='Penalties', color='gray', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"gap={g}" for g in gaps])
    ax.set_ylabel("Score"); ax.set_title("Fitness Component Breakdown\n(stacked positive, penalties below)", fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Panel C: Displacement only (the fair comparison)
    ax = axes[2]
    disp_vals = [all_decomp[g]["displacement"] for g in gaps]
    bars = ax.bar(x, disp_vals, 0.5, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    for bar, val in zip(bars, disp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+3, f"{val:.1f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f"gap={g}" for g in gaps])
    ax.set_ylabel("Displacement (X units)"); ax.set_title("Pure Displacement\n(The Fair Comparison)", fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    
    fig.suptitle("Reward Paradox Analysis: Why Does gap=5.0 Win?", fontsize=15, fontweight='bold')
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "paradox_decomposition.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  Figure 1: {path1}")
    
    # === FIGURE 2: Trajectory comparison (3 panels side by side) ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, (gap, label, col) in enumerate(zip(gaps, labels, colors)):
        tk = all_tracking[gap]
        cs = tk["combine_step"]
        
        # Top row: COM X position over time
        ax = axes[0, i]
        times = np.arange(len(tk["com0_x"])) * DT
        ax.plot(times, tk["com0_x"], '#e74c3c', linewidth=1.5, label='Body 0', alpha=0.7)
        ax.plot(times, tk["com1_x"], '#3498db', linewidth=1.5, label='Body 1', alpha=0.7)
        ax.plot(times, tk["total_com_x"], 'black', linewidth=2, label='Combined COM')
        if cs:
            ax.axvline(x=cs*DT, color='gold', linewidth=2, linestyle='--',
                       label=f'Combine (t={cs*DT:.1f}s)')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("X Position")
        ax.set_title(f"{label}\nDisp={all_decomp[gap]['displacement']:.1f}", fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        
        # Bottom row: COM distance over time
        ax = axes[1, i]
        ax.plot(times, tk["com_dist"], col, linewidth=2)
        ax.axhline(y=1.2, color='gold', linewidth=1.5, linestyle='--',
                    label='Combine threshold')
        if cs:
            ax.axvline(x=cs*DT, color='gold', linewidth=2, linestyle='--', alpha=0.5)
            # Shade approach phase
            ax.axvspan(0, cs*DT, alpha=0.1, color='orange', label='Approach phase')
            ax.axvspan(cs*DT, times[-1], alpha=0.1, color='green', label='Locomotion phase')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Inter-body Distance")
        ax.set_title(f"Body Distance | Combine@step={cs}", fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    
    fig.suptitle("Behavior Analysis: Approach Phase vs Locomotion Phase", fontsize=15, fontweight='bold')
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "paradox_trajectories.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Figure 2: {path2}")
    
    # === FIGURE 3: Filmstrip for gap=5.0 (the most interesting) ===
    tk5 = all_tracking[5.0]
    n_frames = len(tk5["traj_frames"])
    key_indices = [0, n_frames//5, 2*n_frames//5, 3*n_frames//5, 4*n_frames//5, n_frames-1]
    cs5 = tk5["combine_step"] or 0
    
    fig, axes = plt.subplots(1, 6, figsize=(30, 5), subplot_kw={'projection': '3d'})
    for ax_i, (ax, ki) in enumerate(zip(axes, key_indices)):
        fp = tk5["traj_frames"][ki]
        bids = tk5["body_ids"]
        for bi, col in [(0, '#e74c3c'), (1, '#3498db')]:
            m = bids == bi
            ax.scatter(fp[m,0], fp[m,2], fp[m,1], c=col, s=8, alpha=0.8)
        t_sec = ki * 10 * DT
        phase = "APPROACH" if ki*10 < cs5 else "LOCOMOTION"
        ax.set_title(f"t={t_sec:.1f}s\n[{phase}]", fontsize=9)
        cx = fp[:,0].mean()
        ax.set_xlim(cx-8, cx+8); ax.set_ylim(-4,4); ax.set_zlim(-3,6)
        ax.view_init(elev=20, azim=-60)
    fig.suptitle(f"Gap=5.0 Filmstrip: Approach -> Combine -> Walk | Combine@step={cs5}",
                 fontsize=14, fontweight='bold')
    path3 = os.path.join(OUTPUT_DIR, "paradox_filmstrip_gap5.png")
    plt.savefig(path3, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Figure 3: {path3}")
    
    # === Save results ===
    results = {
        "analysis": "Dual-Mode Reward Paradox Investigation",
        "decomposed": {str(g): all_decomp[g] for g in gaps},
        "combine_steps": {str(g): all_tracking[g]["combine_step"] for g in gaps},
        "conclusion": "",
    }
    
    # Determine conclusion
    pure_loco = {g: all_decomp[g]["pure_locomotion"] for g in gaps}
    disp_vals = {g: all_decomp[g]["displacement"] for g in gaps}
    best_pure = max(pure_loco, key=pure_loco.get)
    best_disp = max(disp_vals, key=disp_vals.get)
    
    if best_pure != 5.0:
        results["conclusion"] = (
            f"CONFIRMED: Reward hack detected! "
            f"gap={best_disp} has highest displacement ({disp_vals[best_disp]:.1f}) "
            f"and gap={best_pure} has best pure locomotion ({pure_loco[best_pure]:.1f}). "
            f"The gap=5.0 advantage in total fitness was due to inflated approach_reward."
        )
    else:
        results["conclusion"] = (
            f"SURPRISING: gap=5.0 genuinely achieves best pure locomotion ({pure_loco[5.0]:.1f})! "
            f"The longer approach phase may have provided a beneficial 'warm-up' period."
        )
    
    print(f"\n{'='*60}")
    print(f"  CONCLUSION")
    print(f"{'='*60}")
    print(f"  {results['conclusion']}")
    
    log_path = os.path.join(RESULTS_DIR, "paradox_investigation_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Log: {log_path}")
    
    beep()
    print("\n  PARADOX INVESTIGATION COMPLETE!")


if __name__ == "__main__":
    main()
