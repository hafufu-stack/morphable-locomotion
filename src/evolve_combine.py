"""
GPU 3D Combining Robot Simulation — V2.1
==========================================

Two soft-body robots start ADJACENT and COMBINE IMMEDIATELY at step 0.
Evolution focuses on locomotion of the merged body.

V2.1 strategy:
  - Bodies placed with only 0.5 unit gap → combine springs form instantly
  - No approach phase needed → NN fully dedicated to locomotion
  - 10×5×4 = 200 particles per body, 400 total
  - Combine bonus removed (always combined) → fitness = pure displacement

Body: 10×5×4 = 200 particles per robot

Usage:
    python src/evolve_combine.py
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
print(f"  Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# === Physics ===
GRID_X, GRID_Y, GRID_Z = 10, 5, 4
N_PER_BODY = GRID_X * GRID_Y * GRID_Z  # 200
N_BODIES = 2
N_PARTICLES = N_PER_BODY * N_BODIES  # 400
SPACING = 0.35
DT = 0.010
N_STEPS = 600  # 6 seconds of locomotion (no approach phase needed)
SPRING_K = 30.0
SPRING_DAMP = 1.5
DRAG = 0.4
GROUND_Y = -0.5
GROUND_K = 600.0
GRAVITY = -9.8
FRICTION = 3.0
BASE_AMP = 30.0

# === Combining ===
COMBINE_DIST = 1.2    # Distance threshold for forming initial springs
COMBINE_CHECK = 10
MAX_NEW_SPRINGS = 500 # More springs for initial combine

# === GA ===
BATCH_SIZE = 200   # Slightly smaller batch for bigger bodies
N_GENERATIONS = 150
MUTATION_RATE = 0.15
MUTATION_SIGMA = 0.3
ELITE_FRAC = 0.05
TOURNAMENT_SIZE = 5

# === NN ===
HIDDEN_SIZE = 32   # Bigger hidden for more complex behavior
INPUT_SIZE = 7     # [sin, cos, nx, ny, nz, body_id, is_combined]
OUTPUT_SIZE = 3    # [Fx, Fy, Fz]
N_W1 = INPUT_SIZE * HIDDEN_SIZE
N_B1 = HIDDEN_SIZE
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE
N_B2 = OUTPUT_SIZE
N_FREQ = 1
N_GENES = N_W1 + N_B1 + N_W2 + N_B2 + N_FREQ  # 7*32+32+32*3+3+1 = 356

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================================================
#  Build Two-Body Template
# ===================================================

def build_two_bodies():
    """Create two 3D bodies separated along X axis, facing each other."""
    all_pos = np.zeros((N_PARTICLES, 3))
    body_ids = np.zeros(N_PARTICLES, dtype=np.int64)  # 0 or 1
    
    # Body 0: right side of combined body
    idx = 0
    body_width = (GRID_X - 1) * SPACING
    gap = 0.5  # Small gap between bodies → will combine immediately
    
    for gx in range(GRID_X):
        for gy in range(GRID_Y):
            for gz in range(GRID_Z):
                all_pos[idx] = [
                    -(gap/2 + body_width) + gx * SPACING,
                    2.0 + gy * SPACING,
                    gz * SPACING - (GRID_Z - 1) * SPACING / 2,
                ]
                body_ids[idx] = 0
                idx += 1
    
    # Body 1: left side, mirrored, nearly touching body 0
    for gx in range(GRID_X):
        for gy in range(GRID_Y):
            for gz in range(GRID_Z):
                all_pos[idx] = [
                    (gap/2 + body_width) - gx * SPACING,  # Mirror
                    2.0 + gy * SPACING,
                    gz * SPACING - (GRID_Z - 1) * SPACING / 2,
                ]
                body_ids[idx] = 1
                idx += 1
    
    # Build springs within each body
    springs_a = []
    springs_b = []
    rest_lengths = []
    
    for body_id in range(N_BODIES):
        mask = np.where(body_ids == body_id)[0]
        body_pos = all_pos[mask]
        tri = Delaunay(body_pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = simplex[i], simplex[j]
                    # Map back to global indices
                    ga, gb = mask[a], mask[b]
                    edge = (min(ga, gb), max(ga, gb))
                    edges.add(edge)
        for a_g, b_g in edges:
            springs_a.append(a_g)
            springs_b.append(b_g)
            rest_lengths.append(np.linalg.norm(all_pos[a_g] - all_pos[b_g]))
    
    # Normalize positions per-body for NN input
    norm_pos = np.zeros((N_PARTICLES, 3))
    for body_id in range(N_BODIES):
        mask = body_ids == body_id
        for dim in range(3):
            vmin = all_pos[mask, dim].min()
            vmax = all_pos[mask, dim].max()
            rng = vmax - vmin + 1e-8
            norm_pos[mask, dim] = 2 * (all_pos[mask, dim] - vmin) / rng - 1
    
    return (all_pos, norm_pos, body_ids,
            np.array(springs_a), np.array(springs_b), np.array(rest_lengths))


# ===================================================
#  GPU Simulation with Dynamic Combining
# ===================================================

@torch.no_grad()
def simulate_batch(genomes, rest_pos_t, norm_pos_t, body_ids_t,
                   spring_a_t, spring_b_t, rest_lengths_t):
    """Simulate ALL genomes in parallel. Returns fitness (B,) and displacement (B,)."""
    B = genomes.shape[0]
    N = N_PARTICLES
    
    pos = rest_pos_t.unsqueeze(0).expand(B, -1, -1).clone()
    vel = torch.zeros(B, N, 3, device=DEVICE)
    
    # Unpack NN
    idx = 0
    W1 = genomes[:, idx:idx + N_W1].reshape(B, INPUT_SIZE, HIDDEN_SIZE); idx += N_W1
    b1 = genomes[:, idx:idx + N_B1].unsqueeze(1); idx += N_B1
    W2 = genomes[:, idx:idx + N_W2].reshape(B, HIDDEN_SIZE, OUTPUT_SIZE); idx += N_W2
    b2 = genomes[:, idx:idx + N_B2].unsqueeze(1); idx += N_B2
    freq = genomes[:, idx].abs()
    
    start_com_x = pos[:, :, 0].mean(dim=1)
    
    # Body IDs and initial norm positions
    body_id_input = body_ids_t.float().unsqueeze(0).unsqueeze(2).expand(B, N, 1)
    norm_inp = norm_pos_t.unsqueeze(0).expand(B, -1, -1)
    
    # Dynamic springs: start with internal springs, add combining springs later
    # We use the same spring set for ALL batch items (simplification)
    cur_spring_a = spring_a_t.clone()
    cur_spring_b = spring_b_t.clone()
    cur_rest_len = rest_lengths_t.clone()
    
    combined = torch.zeros(B, 1, 1, device=DEVICE)  # 0/1 flag per batch
    total_energy = torch.zeros(B, device=DEVICE)
    
    # Pre-compute body masks for combining check
    body0_mask = (body_ids_t == 0)  # (N,)
    body1_mask = (body_ids_t == 1)  # (N,)
    body0_indices = body0_mask.nonzero(as_tuple=True)[0]
    body1_indices = body1_mask.nonzero(as_tuple=True)[0]
    
    combine_done = False
    
    for step in range(N_STEPS):
        t = step * DT
        
        # === Combine at step 0 (or first check) ===
        if not combine_done and step % COMBINE_CHECK == 0:
            p0 = pos[0, body0_indices]  # (N0, 3)
            p1 = pos[0, body1_indices]  # (N1, 3)
            dists = torch.cdist(p0, p1)  # (N0, N1)
            close = (dists < COMBINE_DIST).nonzero(as_tuple=False)
            
            if close.shape[0] > 0:
                n_new = min(close.shape[0], MAX_NEW_SPRINGS)
                new_a_local = close[:n_new, 0]
                new_b_local = close[:n_new, 1]
                new_a_global = body0_indices[new_a_local]
                new_b_global = body1_indices[new_b_local]
                new_rest = dists[new_a_local, new_b_local]
                
                cur_spring_a = torch.cat([cur_spring_a, new_a_global])
                cur_spring_b = torch.cat([cur_spring_b, new_b_global])
                cur_rest_len = torch.cat([cur_rest_len, new_rest])
                
                combined = torch.ones(B, 1, 1, device=DEVICE)
                combine_done = True
        
        # === NN Forward Pass ===
        sin_t = torch.sin(2 * np.pi * freq * t).reshape(B, 1, 1).expand(B, N, 1)
        cos_t = torch.cos(2 * np.pi * freq * t).reshape(B, 1, 1).expand(B, N, 1)
        combined_input = combined.expand(B, N, 1)
        
        nn_input = torch.cat([sin_t, cos_t, norm_inp, body_id_input, combined_input], dim=2)
        
        hidden = torch.tanh(torch.bmm(nn_input, W1) + b1)
        output = torch.tanh(torch.bmm(hidden, W2) + b2)
        
        on_ground = (pos[:, :, 1] < GROUND_Y + 0.3).float()
        gc = 0.5 + 1.0 * on_ground
        
        ext = torch.zeros(B, N, 3, device=DEVICE)
        ext[:, :, 0] = BASE_AMP * output[:, :, 0] * gc
        ext[:, :, 1] = BASE_AMP * torch.clamp(output[:, :, 1], min=0) * gc
        ext[:, :, 2] = BASE_AMP * output[:, :, 2] * gc * 0.5
        
        # Energy tracking (NN output only)
        total_energy += (ext ** 2).sum(dim=(1, 2))
        
        # === Physics ===
        forces = torch.zeros(B, N, 3, device=DEVICE)
        forces[:, :, 1] += GRAVITY
        
        S = cur_spring_a.shape[0]
        pa = pos[:, cur_spring_a]
        pb = pos[:, cur_spring_b]
        va = vel[:, cur_spring_a]
        vb = vel[:, cur_spring_b]
        
        diff = pb - pa
        dist = torch.norm(diff, dim=2, keepdim=True).clamp(min=1e-8)
        direction = diff / dist
        rest = cur_rest_len.unsqueeze(0).unsqueeze(2)
        stretch = dist - rest
        rel_v = vb - va
        vel_along = (rel_v * direction).sum(dim=2, keepdim=True)
        
        f_s = SPRING_K * stretch * direction
        f_d = SPRING_DAMP * vel_along * direction
        ft = f_s + f_d
        
        forces.scatter_add_(1, cur_spring_a.unsqueeze(0).unsqueeze(2).expand(B, -1, 3), ft)
        forces.scatter_add_(1, cur_spring_b.unsqueeze(0).unsqueeze(2).expand(B, -1, 3), -ft)
        
        pen = (GROUND_Y - pos[:, :, 1]).clamp(min=0)
        forces[:, :, 1] += GROUND_K * pen
        bl = (pos[:, :, 1] < GROUND_Y).float()
        forces[:, :, 0] -= FRICTION * vel[:, :, 0] * bl
        forces[:, :, 2] -= FRICTION * vel[:, :, 2] * bl
        
        forces -= DRAG * vel
        forces += ext
        
        vel += forces * DT
        pos += vel * DT
    
    # Fitness: combined body forward displacement
    end_com_x = pos[:, :, 0].mean(dim=1)
    displacement = end_com_x - start_com_x
    
    drift_z = pos[:, :, 2].mean(dim=1).abs()
    spread = pos.max(dim=1).values - pos.min(dim=1).values
    spread_pen = ((spread - 8.0).clamp(min=0) * 1.5).sum(dim=1)
    below = (pos[:, :, 1] < GROUND_Y - 1.0).float().sum(dim=1) * 0.2
    
    max_e = N_PARTICLES * N_STEPS * (BASE_AMP * 1.5) ** 2 * 3
    energy_pen = 1.0 * (total_energy / max_e) * 100
    
    # Bonus: simple proximity reward (mostly redundant since always combined)
    com0 = pos[:, body0_mask].mean(dim=1)  # (B, 3)
    com1 = pos[:, body1_mask].mean(dim=1)  # (B, 3)
    merge_dist = torch.norm(com0 - com1, dim=1)
    cohesion_bonus = torch.clamp(3.0 - merge_dist, min=0) * 2.0  # Keep body together
    
    fitness = displacement - drift_z - spread_pen - below - energy_pen + cohesion_bonus
    
    return fitness, displacement


# ===================================================
#  GA
# ===================================================

def evolve():
    print("=" * 60)
    print("  GPU 3D COMBINING ROBOT Evolution")
    print(f"  Body: {GRID_X}×{GRID_Y}×{GRID_Z} = {N_PER_BODY} particles × {N_BODIES} = {N_PARTICLES} total")
    print(f"  NN: Input({INPUT_SIZE}) -> Hidden({HIDDEN_SIZE}) -> Output({OUTPUT_SIZE})")
    print(f"  Genome: {N_GENES} params | Pop: {BATCH_SIZE}")
    print(f"  Combine threshold: {COMBINE_DIST} | Instant combine (gap=0.5)")
    print("=" * 60)
    
    data = build_two_bodies()
    all_pos, norm_pos, body_ids, sa, sb, rl = data
    n_springs = len(sa)
    print(f"  Internal springs: {n_springs}")
    
    rest_pos_t = torch.tensor(all_pos, dtype=torch.float32, device=DEVICE)
    norm_pos_t = torch.tensor(norm_pos, dtype=torch.float32, device=DEVICE)
    body_ids_t = torch.tensor(body_ids, dtype=torch.long, device=DEVICE)
    spring_a_t = torch.tensor(sa, dtype=torch.long, device=DEVICE)
    spring_b_t = torch.tensor(sb, dtype=torch.long, device=DEVICE)
    rest_len_t = torch.tensor(rl, dtype=torch.float32, device=DEVICE)
    
    # Init population
    s1 = np.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE))
    s2 = np.sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE))
    pop = torch.randn(BATCH_SIZE, N_GENES, device=DEVICE) * 0.3
    pop[:, :N_W1] *= s1 / 0.3
    pop[:, N_W1:N_W1 + N_B1] = 0
    pop[:, N_W1 + N_B1:N_W1 + N_B1 + N_W2] *= s2 / 0.3
    pop[:, N_W1 + N_B1 + N_W2:N_W1 + N_B1 + N_W2 + N_B2] = 0
    pop[:, -1] = torch.empty(BATCH_SIZE, device=DEVICE).uniform_(0.5, 3.0)
    
    pop_fit = torch.full((BATCH_SIZE,), float('-inf'), device=DEVICE)
    history = {"best": [], "avg": []}
    n_elite = max(1, int(BATCH_SIZE * ELITE_FRAC))
    stag = 0
    prev_best = -float('inf')
    t0 = time.time()
    
    for gen in range(N_GENERATIONS):
        gen_t0 = time.time()
        
        uneval = pop_fit == float('-inf')
        if uneval.any():
            fit_new, _ = simulate_batch(
                pop[uneval], rest_pos_t, norm_pos_t, body_ids_t,
                spring_a_t, spring_b_t, rest_len_t)
            pop_fit[uneval] = fit_new
        
        order = pop_fit.argsort(descending=True)
        pop = pop[order]; pop_fit = pop_fit[order]
        
        bf = pop_fit[0].item()
        af = pop_fit.mean().item()
        history["best"].append(bf)
        history["avg"].append(af)
        
        if bf > prev_best + 0.1:
            stag = 0; prev_best = bf
        else:
            stag += 1
        
        gen_dt = time.time() - gen_t0
        if gen % 10 == 0 or gen == N_GENERATIONS - 1:
            freq = abs(pop[0, -1].item())
            print(f"  Gen {gen:3d}/{N_GENERATIONS}: best={bf:+8.2f}  "
                  f"avg={af:+8.2f}  freq={freq:.2f}Hz  stag={stag}  [{gen_dt:.1f}s]")
        
        sig = MUTATION_SIGMA * (1.5 if stag > 20 else 1.0) * (2.0 if stag > 50 else 1.0)
        rate = MUTATION_RATE * (1.5 if stag > 20 else 1.0) * (2.0 if stag > 50 else 1.0)
        
        new_pop = pop[:n_elite].clone()
        new_fit = pop_fit[:n_elite].clone()
        
        if stag > 30:
            nf = int(BATCH_SIZE * 0.1)
            fresh = torch.randn(nf, N_GENES, device=DEVICE) * 0.3
            fresh[:, :N_W1] *= s1 / 0.3
            fresh[:, N_W1:N_W1+N_B1] = 0
            fresh[:, N_W1+N_B1:N_W1+N_B1+N_W2] *= s2 / 0.3
            fresh[:, N_W1+N_B1+N_W2:N_W1+N_B1+N_W2+N_B2] = 0
            fresh[:, -1] = torch.empty(nf, device=DEVICE).uniform_(0.5, 3.0)
            new_pop = torch.cat([new_pop, fresh])
            new_fit = torch.cat([new_fit, torch.full((nf,), float('-inf'), device=DEVICE)])
        
        nc = BATCH_SIZE - new_pop.shape[0]
        t1 = torch.randint(BATCH_SIZE, (nc, TOURNAMENT_SIZE), device=DEVICE)
        p1_idx = t1[torch.arange(nc, device=DEVICE), pop_fit[t1].argmax(dim=1)]
        t2 = torch.randint(BATCH_SIZE, (nc, TOURNAMENT_SIZE), device=DEVICE)
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
    return pop[0].cpu().numpy(), pop_fit[0].item(), history, total


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(best_genes, best_fit, history, total_time):
    data = build_two_bodies()
    all_pos, norm_pos, body_ids, sa, sb, rl = data
    springs_list = list(zip(sa, sb))
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"GPU 3D Combining Robot: {N_PER_BODY}×{N_BODIES}={N_PARTICLES} particles\n"
                 f"NN({INPUT_SIZE}→{HIDDEN_SIZE}→{OUTPUT_SIZE}), {N_GENES} params | "
                 f"Best={best_fit:.2f} | {total_time/60:.1f} min",
                 fontsize=13, fontweight='bold')
    
    # Panel 1: Fitness
    ax1 = fig.add_subplot(2, 2, 1)
    gens = range(len(history["best"]))
    ax1.fill_between(gens, history["avg"], history["best"], alpha=0.15, color='#e74c3c')
    ax1.plot(gens, history["best"], '#e74c3c', linewidth=2, label='Best')
    ax1.plot(gens, history["avg"], '#f39c12', linewidth=1.5, alpha=0.6, label='Average')
    ax1.set_xlabel("Gen"); ax1.set_ylabel("Fitness")
    ax1.set_title("Combining Robot Fitness", fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    
    # Panel 2: Initial config (two bodies, colored differently)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    for bi, col in [(0, '#e74c3c'), (1, '#3498db')]:
        mask = body_ids == bi
        ax2.scatter(all_pos[mask, 0], all_pos[mask, 2], all_pos[mask, 1],
                    c=col, s=30, alpha=0.8, edgecolors='black', linewidth=0.3)
    lines = [[all_pos[a], all_pos[b]] for a, b in springs_list]
    lc = Line3DCollection(lines, colors='gray', alpha=0.15, linewidths=0.3)
    ax2.add_collection3d(lc)
    ax2.set_xlabel('X'); ax2.set_ylabel('Z'); ax2.set_zlabel('Y')
    ax2.set_title("Two Bodies (start)", fontweight='bold')
    
    # Panel 3: Replay best genome
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    
    rest_pos_t = torch.tensor(all_pos, dtype=torch.float32, device=DEVICE)
    norm_pos_t = torch.tensor(norm_pos, dtype=torch.float32, device=DEVICE)
    body_ids_t = torch.tensor(body_ids, dtype=torch.long, device=DEVICE)
    spring_a_t = torch.tensor(sa, dtype=torch.long, device=DEVICE)
    spring_b_t = torch.tensor(sb, dtype=torch.long, device=DEVICE)
    rest_len_t = torch.tensor(rl, dtype=torch.float32, device=DEVICE)
    
    genes_t = torch.tensor(best_genes, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    B = 1; N = N_PARTICLES
    pos = rest_pos_t.unsqueeze(0).clone()
    vel_t = torch.zeros(B, N, 3, device=DEVICE)
    
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
    combine_done = False
    combine_step = None
    n_new_springs = 0
    traj = []
    d_hist = []  # Distance history for Panel 4 (collect during same replay)
    
    with torch.no_grad():
        for step in range(N_STEPS):
            t = step * DT
            if not combine_done and step % COMBINE_CHECK == 0:
                p0 = pos[0, body0_idx]; p1 = pos[0, body1_idx]
                ds = torch.cdist(p0, p1)
                cl = (ds < COMBINE_DIST).nonzero(as_tuple=False)
                if cl.shape[0] > 0:
                    nn = min(cl.shape[0], MAX_NEW_SPRINGS)
                    na = body0_idx[cl[:nn, 0]]; nb = body1_idx[cl[:nn, 1]]
                    nr = ds[cl[:nn, 0], cl[:nn, 1]]
                    cur_sa = torch.cat([cur_sa, na]); cur_sb = torch.cat([cur_sb, nb])
                    cur_rl = torch.cat([cur_rl, nr])
                    combined_flag = torch.ones(B, 1, 1, device=DEVICE)
                    combine_done = True; combine_step = step; n_new_springs = nn
            
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
            # V2.1: no direction forcing (bodies already combined)
            
            forces = torch.zeros(B,N,3, device=DEVICE); forces[:,:,1] += GRAVITY
            pa = pos[:,cur_sa]; pb = pos[:,cur_sb]
            diff = pb-pa; dist = torch.norm(diff,dim=2,keepdim=True).clamp(min=1e-8)
            direction = diff/dist; rest = cur_rl.unsqueeze(0).unsqueeze(2)
            stretch = dist-rest; rv = vel_t[:,cur_sb]-vel_t[:,cur_sa]
            va = (rv*direction).sum(dim=2,keepdim=True)
            ft = SPRING_K*stretch*direction + SPRING_DAMP*va*direction
            forces.scatter_add_(1, cur_sa.unsqueeze(0).unsqueeze(2).expand(B,-1,3), ft)
            forces.scatter_add_(1, cur_sb.unsqueeze(0).unsqueeze(2).expand(B,-1,3), -ft)
            pen = (GROUND_Y-pos[:,:,1]).clamp(min=0); forces[:,:,1] += GROUND_K*pen
            bl = (pos[:,:,1]<GROUND_Y).float()
            forces[:,:,0] -= FRICTION*vel_t[:,:,0]*bl
            forces[:,:,2] -= FRICTION*vel_t[:,:,2]*bl
            forces -= DRAG*vel_t; forces += ext
            vel_t += forces*DT; pos += vel_t*DT
            
            if step % 30 == 0:
                com = pos[0].mean(dim=0).cpu().numpy()
                traj.append(com)
            # Collect distance history for Panel 4
            if step % 5 == 0:
                c0 = pos[0, body0_m].mean(dim=0)
                c1 = pos[0, body1_m].mean(dim=0)
                d_hist.append(torch.norm(c0 - c1).item())
    
    pos_np = pos[0].cpu().numpy()
    traj = np.array(traj)
    
    for bi, col in [(0, '#e74c3c'), (1, '#3498db')]:
        mask = body_ids == bi
        ax3.scatter(pos_np[mask, 0], pos_np[mask, 2], pos_np[mask, 1],
                    c=col, s=30, alpha=0.8, edgecolors='black', linewidth=0.3)
    
    # Draw combining springs in gold
    all_springs = list(zip(cur_sa.cpu().numpy(), cur_sb.cpu().numpy()))
    n_orig = len(springs_list)
    if len(all_springs) > n_orig:
        new_spring_lines = [[pos_np[a], pos_np[b]] for a, b in all_springs[n_orig:]]
        lc_new = Line3DCollection(new_spring_lines, colors='gold', alpha=0.6, linewidths=1.5)
        ax3.add_collection3d(lc_new)
    
    ax3.plot(traj[:, 0], traj[:, 2], traj[:, 1], 'k--', linewidth=1.5, alpha=0.5)
    end_x = pos_np[:, 0].mean()
    combine_info = f"Combined @step {combine_step} (+{n_new_springs} springs)" if combine_done else "Not combined"
    ax3.set_xlabel('X'); ax3.set_ylabel('Z'); ax3.set_zlabel('Y')
    ax3.set_title(f"After {N_STEPS*DT:.1f}s: X={end_x:.1f}\n{combine_info}", fontweight='bold')
    
    # Panel 4: Distance between bodies over time (from Panel 3 replay data)
    ax4 = fig.add_subplot(2, 2, 4)
    times = np.arange(len(d_hist)) * 5 * DT
    ax4.plot(times, d_hist, '#e74c3c', linewidth=2)
    ax4.axhline(y=COMBINE_DIST, color='gold', linestyle='--', linewidth=1.5,
                label=f'Combine threshold ({COMBINE_DIST})')
    # V2.1: instant combine, no start delay line needed
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Inter-body COM Distance")
    ax4.set_title("Body Approach & Combination", fontweight='bold')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.2)
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "evolution_combine.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path, end_x, combine_step, n_new_springs


def main():
    best_genes, best_fit, history, total_time = evolve()
    
    # Save log FIRST (before visualization which might crash)
    freq = abs(best_genes[-1])
    results_early = {
        "experiment": "GPU 3D Combining Robot",
        "device": str(DEVICE),
        "bodies": {"per_body": N_PER_BODY, "total": N_PARTICLES},
        "nn": {"input": INPUT_SIZE, "hidden": HIDDEN_SIZE, "output": OUTPUT_SIZE,
               "params": N_GENES},
        "results": {"fitness": float(best_fit), "freq": float(freq)},
        "elapsed_min": round(total_time/60, 1),
    }
    log_path = os.path.join(RESULTS_DIR, "evolution_combine_log.json")
    with open(log_path, "w") as f:
        json.dump(results_early, f, indent=2)
    print(f"  Log (pre-viz): {log_path}")
    
    # Clear GPU memory before visualization
    torch.cuda.empty_cache()
    
    try:
        fig_path, disp, cstep, nsprings = visualize(best_genes, best_fit, history, total_time)
    except Exception as e:
        print(f"  ⚠️ Visualization error: {e}")
        import traceback; traceback.print_exc()
        disp = 0; cstep = None; nsprings = 0
    
    print(f"\n{'='*60}")
    print(f"  COMBINING ROBOT RESULTS")
    print(f"{'='*60}")
    print(f"  Bodies: {N_PER_BODY} × {N_BODIES} = {N_PARTICLES}")
    print(f"  NN: {INPUT_SIZE}→{HIDDEN_SIZE}→{OUTPUT_SIZE} ({N_GENES} params)")
    print(f"  Freq: {freq:.3f} Hz")
    print(f"  Fitness: {best_fit:.2f}")
    print(f"  Displacement: {disp:.2f}")
    if cstep:
        print(f"  Combined at step {cstep} ({cstep*DT:.2f}s), +{nsprings} springs")
    else:
        print(f"  Bodies did NOT combine")
    print(f"{'='*60}")
    
    # Update log with viz results
    results_early["results"]["displacement"] = float(disp)
    results_early["results"]["combined_step"] = int(cstep) if cstep else None
    results_early["results"]["new_springs"] = int(nsprings)
    with open(log_path, "w") as f:
        json.dump(results_early, f, indent=2)
    print(f"  Log: {log_path}")
    
    try:
        import winsound
        # Victory fanfare!
        for freq_b in [523, 659, 784, 1047]:
            winsound.Beep(freq_b, 250)
            time.sleep(0.05)
        print("\n  🔔 BEEP! Combining robot experiment complete!")
    except Exception:
        print("\n  Done!")


if __name__ == "__main__":
    main()
