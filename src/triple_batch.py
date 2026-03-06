"""
Triple Experiment Batch
========================
Runs 3 experiments sequentially (autonomous, tethering-safe):
  1. V2.1 Animation (frame-by-frame replay → GIF)
  2. Bone & Muscle (異種間共生: hard+heavy vs soft+light)
  3. Dual-Mode (接近→合体→歩行)

Each experiment saves its own figures and logs.
Beeps on completion of each, victory fanfare at the end.

Usage:
    python src/triple_batch.py
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import Delaunay
import os, time, json, glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANIM_DIR = os.path.join(OUTPUT_DIR, "combine_anim")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANIM_DIR, exist_ok=True)

# === Shared Physics ===
DT = 0.010
GROUND_Y = -0.5
GROUND_K = 600.0
GRAVITY = -9.8
FRICTION = 3.0
BASE_AMP = 30.0
DRAG = 0.4

# === NN ===
INPUT_SIZE = 7
HIDDEN_SIZE = 32
OUTPUT_SIZE = 3
N_W1 = INPUT_SIZE * HIDDEN_SIZE
N_B1 = HIDDEN_SIZE
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE
N_B2 = OUTPUT_SIZE
N_FREQ = 1
N_GENES = N_W1 + N_B1 + N_W2 + N_B2 + N_FREQ


def beep(pattern="done"):
    try:
        import winsound
        if pattern == "done":
            for f in [523, 659, 784]: winsound.Beep(f, 200); time.sleep(0.05)
        elif pattern == "victory":
            for f in [523, 659, 784, 1047, 1047]: winsound.Beep(f, 300); time.sleep(0.1)
    except: pass


def build_bodies(grid_x, grid_y, grid_z, spacing, gap, n_bodies=2,
                 body_offsets=None):
    """Build N bodies with configurable grid and gap."""
    n_per = grid_x * grid_y * grid_z
    n_total = n_per * n_bodies
    all_pos = np.zeros((n_total, 3))
    body_ids = np.zeros(n_total, dtype=np.int64)
    body_width = (grid_x - 1) * spacing
    
    idx = 0
    for bi in range(n_bodies):
        sign = -1 if bi == 0 else 1
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
    
    # Springs within each body
    springs_a, springs_b, rest_lengths = [], [], []
    for bi in range(n_bodies):
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
    
    # Normalize positions
    norm_pos = np.zeros_like(all_pos)
    for bi in range(n_bodies):
        m = body_ids == bi
        for d in range(3):
            vmin, vmax = all_pos[m, d].min(), all_pos[m, d].max()
            norm_pos[m, d] = 2*(all_pos[m, d]-vmin)/(vmax-vmin+1e-8)-1
    
    return (all_pos, norm_pos, body_ids,
            np.array(springs_a), np.array(springs_b), np.array(rest_lengths),
            n_per, n_total)


@torch.no_grad()
def simulate_batch(genomes, rest_pos_t, norm_pos_t, body_ids_t,
                   spring_a_t, spring_b_t, rest_lengths_t,
                   n_particles, n_per_body, n_steps,
                   spring_k=30.0, spring_damp=1.5,
                   combine_dist=1.2, max_new_springs=500,
                   fitness_mode="locomotion",
                   per_body_mass=None):
    """GPU-batched simulation. fitness_mode: 'locomotion' or 'dual_mode'"""
    B = genomes.shape[0]
    N = n_particles
    
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
    
    cur_sa = spring_a_t.clone()
    cur_sb = spring_b_t.clone()
    cur_rl = rest_lengths_t.clone()
    combined = torch.zeros(B, 1, 1, device=DEVICE)
    total_energy = torch.zeros(B, device=DEVICE)
    
    body0_mask = (body_ids_t == 0)
    body1_mask = (body_ids_t == 1)
    body0_idx = body0_mask.nonzero(as_tuple=True)[0]
    body1_idx = body1_mask.nonzero(as_tuple=True)[0]
    
    combine_done = False
    
    # Optional per-body mass (for Bone & Muscle)
    if per_body_mass is not None:
        mass = torch.ones(N, device=DEVICE)
        for bi, m in enumerate(per_body_mass):
            mass[body_ids_t == bi] = m
        inv_mass = 1.0 / mass  # (N,)
    else:
        inv_mass = None
    
    # Track approach distance for dual_mode fitness
    approach_reward = torch.zeros(B, device=DEVICE)
    
    for step in range(n_steps):
        t = step * DT
        
        # Combine check
        if not combine_done and step % 10 == 0:
            p0 = pos[0, body0_idx]
            p1 = pos[0, body1_idx]
            dists = torch.cdist(p0, p1)
            close = (dists < combine_dist).nonzero(as_tuple=False)
            if close.shape[0] > 0:
                n_new = min(close.shape[0], max_new_springs)
                na = body0_idx[close[:n_new, 0]]
                nb = body1_idx[close[:n_new, 1]]
                nr = dists[close[:n_new, 0], close[:n_new, 1]]
                cur_sa = torch.cat([cur_sa, na])
                cur_sb = torch.cat([cur_sb, nb])
                cur_rl = torch.cat([cur_rl, nr])
                combined = torch.ones(B, 1, 1, device=DEVICE)
                combine_done = True
        
        # For dual_mode: track how close bodies get (approach reward)
        if fitness_mode == "dual_mode" and not combine_done and step % 20 == 0:
            c0 = pos[:, body0_mask].mean(dim=1)
            c1 = pos[:, body1_mask].mean(dim=1)
            approach_reward += torch.clamp(5.0 - torch.norm(c0-c1, dim=1), min=0)
        
        # NN
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
        
        # Physics
        forces = torch.zeros(B, N, 3, device=DEVICE)
        forces[:,:,1] += GRAVITY
        
        # Per-body mass affects gravity
        if inv_mass is not None:
            forces[:,:,1] = forces[:,:,1] * mass.unsqueeze(0)  # heavier = more gravity
        
        pa = pos[:, cur_sa]; pb = pos[:, cur_sb]
        diff = pb - pa
        dist = torch.norm(diff, dim=2, keepdim=True).clamp(min=1e-8)
        direction = diff / dist
        rest = cur_rl.unsqueeze(0).unsqueeze(2)
        stretch = dist - rest
        rv = vel[:, cur_sb] - vel[:, cur_sa]
        va = (rv * direction).sum(dim=2, keepdim=True)
        ft = spring_k * stretch * direction + spring_damp * va * direction
        forces.scatter_add_(1, cur_sa.unsqueeze(0).unsqueeze(2).expand(B,-1,3), ft)
        forces.scatter_add_(1, cur_sb.unsqueeze(0).unsqueeze(2).expand(B,-1,3), -ft)
        
        pen = (GROUND_Y - pos[:,:,1]).clamp(min=0)
        forces[:,:,1] += GROUND_K * pen
        bl = (pos[:,:,1] < GROUND_Y).float()
        forces[:,:,0] -= FRICTION * vel[:,:,0] * bl
        forces[:,:,2] -= FRICTION * vel[:,:,2] * bl
        forces -= DRAG * vel
        forces += ext
        
        if inv_mass is not None:
            vel += forces * DT * inv_mass.unsqueeze(0).unsqueeze(2)
        else:
            vel += forces * DT
        pos += vel * DT
    
    # Fitness
    end_com_x = pos[:,:,0].mean(dim=1)
    displacement = end_com_x - start_com_x
    drift_z = pos[:,:,2].mean(dim=1).abs()
    spread = pos.max(dim=1).values - pos.min(dim=1).values
    spread_pen = ((spread - 8.0).clamp(min=0)*1.5).sum(dim=1)
    below = (pos[:,:,1] < GROUND_Y-1.0).float().sum(dim=1)*0.2
    max_e = N * n_steps * (BASE_AMP*1.5)**2 * 3
    energy_pen = 1.0 * (total_energy/max_e) * 100
    
    com0 = pos[:, body0_mask].mean(dim=1)
    com1 = pos[:, body1_mask].mean(dim=1)
    merge_dist = torch.norm(com0-com1, dim=1)
    cohesion = torch.clamp(3.0 - merge_dist, min=0) * 2.0
    
    if fitness_mode == "dual_mode":
        combine_bonus = combined.squeeze() * 30.0
        fitness = displacement - drift_z - spread_pen - below - energy_pen + cohesion + approach_reward + combine_bonus
    else:
        fitness = displacement - drift_z - spread_pen - below - energy_pen + cohesion
    
    return fitness, displacement


def evolve_generic(n_steps, grid_x, grid_y, grid_z, spacing, gap,
                   n_gens=150, pop_size=200, spring_k=30.0, spring_damp=1.5,
                   combine_dist=1.2, max_new_springs=500,
                   fitness_mode="locomotion", per_body_mass=None,
                   mutation_rate=0.15, mutation_sigma=0.3,
                   label="Experiment"):
    """Generic evolution loop."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Grid: {grid_x}x{grid_y}x{grid_z} | Gap: {gap} | Steps: {n_steps}")
    print(f"  Spring K: {spring_k} | Combine dist: {combine_dist}")
    if per_body_mass: print(f"  Per-body mass: {per_body_mass}")
    print(f"  Fitness mode: {fitness_mode}")
    print(f"{'='*60}")
    
    data = build_bodies(grid_x, grid_y, grid_z, spacing, gap)
    all_pos, norm_pos, body_ids, sa, sb, rl, n_per, n_total = data
    print(f"  Particles: {n_per} x 2 = {n_total} | Springs: {len(sa)}")
    
    rest_pos_t = torch.tensor(all_pos, dtype=torch.float32, device=DEVICE)
    norm_pos_t = torch.tensor(norm_pos, dtype=torch.float32, device=DEVICE)
    body_ids_t = torch.tensor(body_ids, dtype=torch.long, device=DEVICE)
    spring_a_t = torch.tensor(sa, dtype=torch.long, device=DEVICE)
    spring_b_t = torch.tensor(sb, dtype=torch.long, device=DEVICE)
    rest_len_t = torch.tensor(rl, dtype=torch.float32, device=DEVICE)
    
    # Init population
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
            f, d = simulate_batch(pop[idxn], rest_pos_t, norm_pos_t, body_ids_t,
                                  spring_a_t, spring_b_t, rest_len_t,
                                  n_total, n_per, n_steps,
                                  spring_k=spring_k, spring_damp=spring_damp,
                                  combine_dist=combine_dist,
                                  max_new_springs=max_new_springs,
                                  fitness_mode=fitness_mode,
                                  per_body_mass=per_body_mass)
            pop_fit[idxn] = f
        
        order = pop_fit.argsort(descending=True)
        pop = pop[order]; pop_fit = pop_fit[order]
        history["best"].append(pop_fit[0].item())
        history["avg"].append(pop_fit.mean().item())
        
        if gen % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Gen {gen:3d}/{n_gens}: best={pop_fit[0].item():+8.2f}  "
                  f"avg={pop_fit.mean().item():+8.2f}  [{elapsed/(gen+1):.1f}s/gen]")
        
        ne = max(2, int(pop_size * 0.05))
        rate = mutation_rate; sig = mutation_sigma
        new_pop = pop[:ne].clone()
        new_fit = pop_fit[:ne].clone()
        
        nf = max(2, int(pop_size * 0.05))
        fresh = torch.randn(nf, N_GENES, device=DEVICE) * 0.3
        fresh[:, :N_W1] *= s1/0.3
        fresh[:, -1] = torch.empty(nf, device=DEVICE).uniform_(0.5, 3.0)
        new_pop = torch.cat([new_pop, fresh])
        new_fit = torch.cat([new_fit, torch.full((nf,), float('-inf'), device=DEVICE)])
        
        nc = pop_size - new_pop.shape[0]
        t1 = torch.randint(pop_size, (nc, 5), device=DEVICE)
        p1_idx = t1[torch.arange(nc, device=DEVICE), pop_fit[t1].argmax(dim=1)]
        t2 = torch.randint(pop_size, (nc, 5), device=DEVICE)
        p2_idx = t2[torch.arange(nc, device=DEVICE), pop_fit[t2].argmax(dim=1)]
        mask = torch.rand(nc, N_GENES, device=DEVICE) < 0.5
        children = torch.where(mask, pop[p1_idx], pop[p2_idx])
        mut = torch.rand(nc, N_GENES, device=DEVICE) < rate
        children += torch.randn(nc, N_GENES, device=DEVICE) * sig * mut.float()
        new_pop = torch.cat([new_pop, children])
        new_fit = torch.cat([new_fit, torch.full((nc,), float('-inf'), device=DEVICE)])
        pop = new_pop; pop_fit = new_fit
    
    total = time.time() - t0
    print(f"\n  Total: {total/60:.1f} min | Best: {pop_fit[0].item():+.2f}")
    return pop[0].cpu().numpy(), pop_fit[0].item(), history, total, data


def replay_best(best_genes, data, n_steps, spring_k=30.0, spring_damp=1.5,
                combine_dist=1.2, max_new_springs=500, per_body_mass=None):
    """Replay best genome, return trajectory."""
    all_pos, norm_pos, body_ids, sa, sb, rl, n_per, n_total = data
    N = n_total
    rest_pos_t = torch.tensor(all_pos, dtype=torch.float32, device=DEVICE)
    norm_pos_t = torch.tensor(norm_pos, dtype=torch.float32, device=DEVICE)
    body_ids_t = torch.tensor(body_ids, dtype=torch.long, device=DEVICE)
    spring_a_t = torch.tensor(sa, dtype=torch.long, device=DEVICE)
    spring_b_t = torch.tensor(sb, dtype=torch.long, device=DEVICE)
    rest_len_t = torch.tensor(rl, dtype=torch.float32, device=DEVICE)
    
    genes_t = torch.tensor(best_genes, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    B = 1
    pos = rest_pos_t.unsqueeze(0).clone()
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
    combine_done = False; combine_step = None; n_new = 0
    
    if per_body_mass is not None:
        mass = torch.ones(N, device=DEVICE)
        for bi, m in enumerate(per_body_mass):
            mass[body_ids_t == bi] = m
        inv_mass = 1.0 / mass
    else:
        inv_mass = None
        mass = None
    
    traj = []
    
    with torch.no_grad():
        for step in range(n_steps):
            t = step * DT
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
                    combine_done = True; combine_step = step; n_new = nn
            
            sin_t = torch.sin(2*np.pi*freq_v*t).reshape(B,1,1).expand(B,N,1)
            cos_t = torch.cos(2*np.pi*freq_v*t).reshape(B,1,1).expand(B,N,1)
            nn_in = torch.cat([sin_t, cos_t, norm_inp, body_id_inp,
                               combined_flag.expand(B,N,1)], dim=2)
            hid = torch.tanh(torch.bmm(nn_in, W1) + b1g)
            out = torch.tanh(torch.bmm(hid, W2) + b2g)
            og = (pos[:,:,1] < GROUND_Y+0.3).float()
            gc = 0.5 + og
            ext = torch.zeros(B,N,3, device=DEVICE)
            ext[:,:,0] = BASE_AMP * out[:,:,0] * gc
            ext[:,:,1] = BASE_AMP * torch.clamp(out[:,:,1], min=0) * gc
            ext[:,:,2] = BASE_AMP * out[:,:,2] * gc * 0.5
            
            forces = torch.zeros(B,N,3, device=DEVICE)
            forces[:,:,1] += GRAVITY
            if mass is not None:
                forces[:,:,1] = forces[:,:,1] * mass.unsqueeze(0)
            
            pa = pos[:,cur_sa]; pb = pos[:,cur_sb]
            diff = pb-pa; dist = torch.norm(diff,dim=2,keepdim=True).clamp(min=1e-8)
            direction = diff/dist; rest = cur_rl.unsqueeze(0).unsqueeze(2)
            stretch = dist-rest; rv = vel[:,cur_sb]-vel[:,cur_sa]
            va_s = (rv*direction).sum(dim=2,keepdim=True)
            ft = spring_k*stretch*direction + spring_damp*va_s*direction
            forces.scatter_add_(1, cur_sa.unsqueeze(0).unsqueeze(2).expand(B,-1,3), ft)
            forces.scatter_add_(1, cur_sb.unsqueeze(0).unsqueeze(2).expand(B,-1,3), -ft)
            pen = (GROUND_Y-pos[:,:,1]).clamp(min=0); forces[:,:,1] += GROUND_K*pen
            bl = (pos[:,:,1] < GROUND_Y).float()
            forces[:,:,0] -= FRICTION*vel[:,:,0]*bl
            forces[:,:,2] -= FRICTION*vel[:,:,2]*bl
            forces -= DRAG*vel; forces += ext
            
            if inv_mass is not None:
                vel += forces * DT * inv_mass.unsqueeze(0).unsqueeze(2)
            else:
                vel += forces * DT
            pos += vel * DT
            
            if step % 5 == 0:
                traj.append(pos[0].cpu().numpy().copy())
    
    return traj, body_ids, combine_step, n_new


# ===================================================
#  EXPERIMENT 1: Animation GIF
# ===================================================
def experiment_1_animation():
    print("\n" + "🎬"*30)
    print("  EXPERIMENT 1: V2.1 Animation")
    print("🎬"*30)
    
    grid_x, grid_y, grid_z = 10, 5, 4
    spacing = 0.35; gap = 0.5; n_steps = 600
    
    # First evolve
    best_genes, best_fit, history, total, data = evolve_generic(
        n_steps=n_steps, grid_x=grid_x, grid_y=grid_y, grid_z=grid_z,
        spacing=spacing, gap=gap, n_gens=150, pop_size=200,
        label="EXP 1: V2.1 Animation Run")
    
    # Replay
    traj, body_ids, cstep, nsprings = replay_best(
        best_genes, data, n_steps)
    
    # Generate frames
    print(f"  Generating {len(traj)} animation frames...")
    n_frames = len(traj)
    all_pos_init = data[0]
    
    for i, frame_pos in enumerate(traj):
        if i % 3 != 0:  # Every 3rd frame to keep GIF manageable
            continue
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for bi, col in [(0, '#e74c3c'), (1, '#3498db')]:
            m = body_ids == bi
            ax.scatter(frame_pos[m, 0], frame_pos[m, 2], frame_pos[m, 1],
                      c=col, s=15, alpha=0.8, edgecolors='black', linewidth=0.2)
        
        # Ground plane
        x_range = [frame_pos[:,0].min()-1, frame_pos[:,0].max()+1]
        z_range = [frame_pos[:,2].min()-1, frame_pos[:,2].max()+1]
        ax.plot_surface(
            np.array([[x_range[0], x_range[1]], [x_range[0], x_range[1]]]),
            np.array([[z_range[0], z_range[0]], [z_range[1], z_range[1]]]),
            np.array([[GROUND_Y, GROUND_Y], [GROUND_Y, GROUND_Y]]),
            alpha=0.15, color='green')
        
        t_sec = i * 5 * DT
        ax.set_title(f"Combining Robot | t={t_sec:.2f}s | Fitness={best_fit:.1f}",
                     fontweight='bold', fontsize=12)
        ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
        
        # Dynamic view following the body
        cx = frame_pos[:,0].mean()
        ax.set_xlim(cx-5, cx+5)
        ax.set_ylim(-3, 3); ax.set_zlim(-2, 5)
        ax.view_init(elev=20, azim=-60)
        
        plt.savefig(os.path.join(ANIM_DIR, f"frame_{i:04d}.png"), dpi=100,
                    bbox_inches='tight')
        plt.close()
    
    # Compile GIF
    try:
        from PIL import Image
        frames = sorted(glob.glob(os.path.join(ANIM_DIR, "frame_*.png")))
        if frames:
            imgs = [Image.open(f) for f in frames]
            gif_path = os.path.join(OUTPUT_DIR, "combine_animation.gif")
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                        duration=80, loop=0)
            print(f"  ✅ GIF saved: {gif_path}")
            # Also save as static filmstrip
            fig2, axes = plt.subplots(1, 5, figsize=(25, 5), subplot_kw={'projection': '3d'})
            key_frames = [0, len(traj)//4, len(traj)//2, 3*len(traj)//4, len(traj)-1]
            for ax_i, (ax, kf) in enumerate(zip(axes, key_frames)):
                fp = traj[kf]
                for bi, col in [(0, '#e74c3c'), (1, '#3498db')]:
                    m = body_ids == bi
                    ax.scatter(fp[m,0], fp[m,2], fp[m,1], c=col, s=8, alpha=0.8)
                cx = fp[:,0].mean()
                ax.set_xlim(cx-5, cx+5); ax.set_ylim(-3,3); ax.set_zlim(-2,5)
                ax.set_title(f"t={kf*5*DT:.1f}s", fontsize=10)
                ax.view_init(elev=20, azim=-60)
            fig2.suptitle(f"V2.1 Combining Robot Filmstrip | Best={best_fit:.1f}", fontsize=14, fontweight='bold')
            strip_path = os.path.join(OUTPUT_DIR, "combine_filmstrip.png")
            plt.savefig(strip_path, dpi=150, bbox_inches='tight'); plt.close()
            print(f"  ✅ Filmstrip: {strip_path}")
    except ImportError:
        print("  ⚠️ Pillow not installed, skipping GIF (frames saved as PNGs)")
    
    # Save log
    log = {"experiment": "V2.1 Animation", "fitness": float(best_fit),
           "n_frames": n_frames, "displacement": float(traj[-1][:,0].mean()-traj[0][:,0].mean())}
    with open(os.path.join(RESULTS_DIR, "exp1_animation_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    
    beep("done")
    print("  ✅ Experiment 1 COMPLETE!")
    return best_fit


# ===================================================
#  EXPERIMENT 2: Bone & Muscle (異種間共生)
# ===================================================
def experiment_2_bone_muscle():
    print("\n" + "🦴"*30)
    print("  EXPERIMENT 2: Bone & Muscle (Symbiosis)")
    print("🦴"*30)
    
    grid_x, grid_y, grid_z = 10, 5, 4
    spacing = 0.35; gap = 0.5; n_steps = 600
    
    # Run 3 conditions
    conditions = {
        "A_homogeneous": {
            "label": "Homogeneous (k=30, mass=1.0 each) [control]",
            "spring_k": 30.0, "per_body_mass": None,
        },
        "B_bone_muscle": {
            "label": "Bone+Muscle (k_bone=100, mass=2.0 | k_muscle=10, mass=0.5)",
            "spring_k": 30.0,  # internal springs same, difference is mass
            "per_body_mass": [2.0, 0.5],
        },
        "C_extreme": {
            "label": "Extreme (mass=3.0 | mass=0.3)",
            "spring_k": 30.0,
            "per_body_mass": [3.0, 0.3],
        },
    }
    
    results = {}
    all_histories = {}
    
    for cond_name, cfg in conditions.items():
        print(f"\n  --- Condition {cond_name}: {cfg['label']} ---")
        best_genes, best_fit, history, total, data = evolve_generic(
            n_steps=n_steps, grid_x=grid_x, grid_y=grid_y, grid_z=grid_z,
            spacing=spacing, gap=gap, n_gens=150, pop_size=200,
            spring_k=cfg["spring_k"],
            per_body_mass=cfg["per_body_mass"],
            label=f"EXP 2: {cfg['label']}")
        results[cond_name] = {"fitness": float(best_fit), "time": total/60}
        all_histories[cond_name] = history
        
        # Replay for filmstrip
        traj, bids, cs, ns = replay_best(
            best_genes, data, n_steps,
            spring_k=cfg["spring_k"],
            per_body_mass=cfg["per_body_mass"])
        results[cond_name]["displacement"] = float(traj[-1][:,0].mean()-traj[0][:,0].mean())
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, (name, hist) in enumerate(all_histories.items()):
        ax = axes[i]
        gens = range(len(hist["best"]))
        ax.fill_between(gens, hist["avg"], hist["best"], alpha=0.15, color=colors[i])
        ax.plot(gens, hist["best"], colors[i], linewidth=2, label='Best')
        ax.plot(gens, hist["avg"], colors[i], linewidth=1, alpha=0.5, label='Avg')
        ax.set_title(f"{conditions[name]['label']}\nBest={results[name]['fitness']:.1f}", fontsize=9)
        ax.set_xlabel("Gen"); ax.set_ylabel("Fitness")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    fig.suptitle("Bone & Muscle Experiment: Mass Asymmetry", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bone_muscle_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  ✅ Figure: {path}")
    
    with open(os.path.join(RESULTS_DIR, "exp2_bone_muscle_log.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    beep("done")
    print("  ✅ Experiment 2 COMPLETE!")
    return results


# ===================================================
#  EXPERIMENT 3: Dual-Mode Brain (接近→合体→歩行)
# ===================================================
def experiment_3_dual_mode():
    print("\n" + "🧠"*30)
    print("  EXPERIMENT 3: Dual-Mode Brain (Approach → Combine → Walk)")
    print("🧠"*30)
    
    grid_x, grid_y, grid_z = 10, 5, 4
    spacing = 0.35; n_steps = 800  # Longer for approach + locomotion
    
    # Run 3 gap conditions
    conditions = {
        "A_close": {"gap": 1.0, "label": "Close (gap=1.0)"},
        "B_medium": {"gap": 3.0, "label": "Medium (gap=3.0)"},
        "C_far": {"gap": 5.0, "label": "Far (gap=5.0)"},
    }
    
    results = {}
    all_histories = {}
    
    for cond_name, cfg in conditions.items():
        print(f"\n  --- Condition {cond_name}: {cfg['label']} ---")
        best_genes, best_fit, history, total, data = evolve_generic(
            n_steps=n_steps, grid_x=grid_x, grid_y=grid_y, grid_z=grid_z,
            spacing=spacing, gap=cfg["gap"], n_gens=200, pop_size=200,
            fitness_mode="dual_mode",
            label=f"EXP 3: Dual-Mode {cfg['label']}")
        results[cond_name] = {"fitness": float(best_fit), "time": total/60}
        all_histories[cond_name] = history
        
        # Replay
        traj, bids, cs, ns = replay_best(best_genes, data, n_steps)
        results[cond_name]["displacement"] = float(traj[-1][:,0].mean()-traj[0][:,0].mean())
        results[cond_name]["combined"] = cs is not None
        results[cond_name]["combine_step"] = cs
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#27ae60', '#e67e22', '#8e44ad']
    for i, (name, hist) in enumerate(all_histories.items()):
        ax = axes[i]
        gens = range(len(hist["best"]))
        ax.fill_between(gens, hist["avg"], hist["best"], alpha=0.15, color=colors[i])
        ax.plot(gens, hist["best"], colors[i], linewidth=2, label='Best')
        ax.plot(gens, hist["avg"], colors[i], linewidth=1, alpha=0.5, label='Avg')
        comb = "✅" if results[name].get("combined") else "❌"
        ax.set_title(f"{conditions[name]['label']} {comb}\n"
                     f"Best={results[name]['fitness']:.1f} | Disp={results[name]['displacement']:.1f}",
                     fontsize=9)
        ax.set_xlabel("Gen"); ax.set_ylabel("Fitness")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    fig.suptitle("Dual-Mode Brain: Approach → Combine → Walk", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "dual_mode_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  ✅ Figure: {path}")
    
    with open(os.path.join(RESULTS_DIR, "exp3_dual_mode_log.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    beep("done")
    print("  ✅ Experiment 3 COMPLETE!")
    return results


# ===================================================
#  MAIN: Run all 3 experiments
# ===================================================
if __name__ == "__main__":
    t_start = time.time()
    print("="*60)
    print("  TRIPLE EXPERIMENT BATCH")
    print(f"  Start time: {time.strftime('%H:%M:%S')}")
    print("="*60)
    
    exp_results = {}
    
    try:
        r1 = experiment_1_animation()
        exp_results["exp1_animation"] = {"fitness": r1}
    except Exception as e:
        print(f"  ❌ Experiment 1 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    torch.cuda.empty_cache()
    
    try:
        r2 = experiment_2_bone_muscle()
        exp_results["exp2_bone_muscle"] = r2
    except Exception as e:
        print(f"  ❌ Experiment 2 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    torch.cuda.empty_cache()
    
    try:
        r3 = experiment_3_dual_mode()
        exp_results["exp3_dual_mode"] = r3
    except Exception as e:
        print(f"  ❌ Experiment 3 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ALL 3 EXPERIMENTS COMPLETE!")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  End time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # Save master log
    exp_results["total_time_min"] = total_time / 60
    master_log = os.path.join(RESULTS_DIR, "triple_batch_log.json")
    with open(master_log, "w") as f:
        json.dump(exp_results, f, indent=2)
    print(f"  Master log: {master_log}")
    
    # Victory fanfare
    beep("victory")
    print("\n  🎉🎉🎉 TRIPLE BATCH COMPLETE! 🎉🎉🎉")
