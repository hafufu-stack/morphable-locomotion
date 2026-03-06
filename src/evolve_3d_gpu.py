"""
GPU-Accelerated 3D Soft-Body Locomotion Evolution
===================================================

Uses PyTorch CUDA to run ALL 300 simulations IN PARALLEL on RTX 5080.
Instead of multiprocessing 22 workers × 1 sim each,
we run 300 sims simultaneously as a batch on GPU.

Key GPU optimizations:
  - Body state: (batch, n_particles, 3) tensors on CUDA
  - Spring forces: vectorized per-spring computation across batch
  - NN forward pass: batched matrix multiply for all particles × all sims
  - No Python loops over particles (all vectorized)

Architecture:
  Body: 6×3×3 = 54 particles, ~300 springs (3D Delaunay)
  NN:   Input(5) -> Hidden(24, tanh) -> Output(3)
  GA:   300 individuals × 200 generations, all on GPU

Expected speedup: ~10-50× over CPU version.

Usage:
    python src/evolve_3d_gpu.py
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

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# === Physics Config ===
GRID_X, GRID_Y, GRID_Z = 6, 3, 3
N_PARTICLES = GRID_X * GRID_Y * GRID_Z  # 54
SPACING = 0.5
DT = 0.012
N_STEPS = 500
SPRING_K = 25.0
SPRING_DAMP = 1.2
DRAG = 0.4
GROUND_Y = -0.5
GROUND_K = 600.0
GRAVITY = -9.8
FRICTION = 3.0
BASE_AMP = 35.0

# === GA Config ===
BATCH_SIZE = 300   # Population = batch size (all on GPU at once!)
N_GENERATIONS = 200
MUTATION_RATE = 0.15
MUTATION_SIGMA = 0.3
ELITE_FRAC = 0.05
TOURNAMENT_SIZE = 5

# === NN Config ===
HIDDEN_SIZE = 24
INPUT_SIZE = 5
OUTPUT_SIZE = 3
N_W1 = INPUT_SIZE * HIDDEN_SIZE    # 120
N_B1 = HIDDEN_SIZE                 # 24
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE   # 72
N_B2 = OUTPUT_SIZE                 # 3
N_FREQ = 1                        # 1
N_GENES = N_W1 + N_B1 + N_W2 + N_B2 + N_FREQ  # 220

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================================================
#  Build Body Template (once, on CPU, then move to GPU)
# ===================================================

def build_body_template():
    """Build rest positions and spring topology (shared across all batch items)."""
    rest_pos = np.zeros((N_PARTICLES, 3))
    idx = 0
    for gx in range(GRID_X):
        for gy in range(GRID_Y):
            for gz in range(GRID_Z):
                rest_pos[idx] = [
                    gx * SPACING - (GRID_X - 1) * SPACING / 2,
                    2.0 + gy * SPACING,
                    gz * SPACING - (GRID_Z - 1) * SPACING / 2,
                ]
                idx += 1
    
    # Delaunay springs
    tri = Delaunay(rest_pos)
    edges = set()
    for simplex in tri.simplices:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = simplex[i], simplex[j]
                edges.add((min(a, b), max(a, b)))
    springs = list(edges)
    spring_a = np.array([s[0] for s in springs])
    spring_b = np.array([s[1] for s in springs])
    rest_lengths = np.array([np.linalg.norm(rest_pos[a] - rest_pos[b]) for a, b in springs])
    
    # Normalized positions for NN input
    norm_pos = np.zeros((N_PARTICLES, 3))
    for dim in range(3):
        vmin = rest_pos[:, dim].min()
        vmax = rest_pos[:, dim].max()
        rng = vmax - vmin + 1e-8
        norm_pos[:, dim] = 2 * (rest_pos[:, dim] - vmin) / rng - 1
    
    return (rest_pos, norm_pos, spring_a, spring_b, rest_lengths, len(springs))


# ===================================================
#  GPU Batched Physics Simulation
# ===================================================

@torch.no_grad()
def simulate_batch(genomes_tensor, rest_pos_t, norm_pos_t, spring_a_t, spring_b_t, rest_lengths_t):
    """
    Run N_STEPS of physics for ALL genomes in parallel on GPU.
    
    genomes_tensor: (batch, N_GENES) float32 on CUDA
    Returns: (fitness, displacement) each (batch,)
    """
    B = genomes_tensor.shape[0]
    N = N_PARTICLES
    
    # Initialize body state: (batch, N, 3)
    pos = rest_pos_t.unsqueeze(0).expand(B, -1, -1).clone()  # (B, N, 3)
    vel = torch.zeros(B, N, 3, device=DEVICE)
    
    # Unpack NN weights from genomes: (batch, ...)
    idx = 0
    W1 = genomes_tensor[:, idx:idx + N_W1].reshape(B, INPUT_SIZE, HIDDEN_SIZE)
    idx += N_W1
    b1 = genomes_tensor[:, idx:idx + N_B1].unsqueeze(1)  # (B, 1, H)
    idx += N_B1
    W2 = genomes_tensor[:, idx:idx + N_W2].reshape(B, HIDDEN_SIZE, OUTPUT_SIZE)
    idx += N_W2
    b2 = genomes_tensor[:, idx:idx + N_B2].unsqueeze(1)  # (B, 1, O)
    idx += N_B2
    freq = genomes_tensor[:, idx].abs()  # (B,)
    
    # Start COM
    start_com_x = pos[:, :, 0].mean(dim=1)  # (B,)
    
    # Precompute norm_pos input: (1, N, 3) -> broadcast to (B, N, 3)
    norm_inp = norm_pos_t.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
    
    total_energy = torch.zeros(B, device=DEVICE)
    
    for step in range(N_STEPS):
        t = step * DT
        
        # === NN Forward Pass (fully batched) ===
        # Time signal: (B, 1, 2)
        sin_t = torch.sin(2 * np.pi * freq * t).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        cos_t = torch.cos(2 * np.pi * freq * t).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        
        # Build input: (B, N, 5) = [sin, cos, nx, ny, nz]
        sin_exp = sin_t.expand(B, N, 1)
        cos_exp = cos_t.expand(B, N, 1)
        nn_input = torch.cat([sin_exp, cos_exp, norm_inp], dim=2)  # (B, N, 5)
        
        # Hidden = tanh(input @ W1 + b1): (B, N, H)
        hidden = torch.tanh(torch.bmm(nn_input, W1) + b1)
        # Output = tanh(hidden @ W2 + b2): (B, N, O=3)
        output = torch.tanh(torch.bmm(hidden, W2) + b2)
        
        # Ground contact modulation: (B, N)
        on_ground = (pos[:, :, 1] < GROUND_Y + 0.3).float()
        ground_contact = 0.5 + 1.0 * on_ground  # 0.5 or 1.5
        gc = ground_contact.unsqueeze(2)  # (B, N, 1)
        
        # External forces: (B, N, 3)
        ext_forces = torch.zeros(B, N, 3, device=DEVICE)
        ext_forces[:, :, 0] = BASE_AMP * output[:, :, 0] * ground_contact
        ext_forces[:, :, 1] = BASE_AMP * torch.clamp(output[:, :, 1], min=0) * ground_contact
        ext_forces[:, :, 2] = BASE_AMP * output[:, :, 2] * ground_contact * 0.5
        
        total_energy += (ext_forces ** 2).sum(dim=(1, 2))
        
        # === Physics Step ===
        forces = torch.zeros(B, N, 3, device=DEVICE)
        
        # Gravity
        forces[:, :, 1] += GRAVITY
        
        # Spring forces (vectorized across batch)
        pa = pos[:, spring_a_t]  # (B, S, 3)
        pb = pos[:, spring_b_t]  # (B, S, 3)
        va = vel[:, spring_a_t]
        vb = vel[:, spring_b_t]
        
        diff = pb - pa  # (B, S, 3)
        dist = torch.norm(diff, dim=2, keepdim=True).clamp(min=1e-8)  # (B, S, 1)
        direction = diff / dist  # (B, S, 3)
        rest_len = rest_lengths_t.unsqueeze(0).unsqueeze(2)  # (1, S, 1)
        stretch = dist - rest_len  # (B, S, 1)
        
        rel_vel = vb - va  # (B, S, 3)
        vel_along = (rel_vel * direction).sum(dim=2, keepdim=True)  # (B, S, 1)
        
        f_spring = SPRING_K * stretch * direction
        f_damp = SPRING_DAMP * vel_along * direction
        f_total = f_spring + f_damp  # (B, S, 3)
        
        # Scatter-add spring forces to particles
        forces.scatter_add_(1, spring_a_t.unsqueeze(0).unsqueeze(2).expand(B, -1, 3), f_total)
        forces.scatter_add_(1, spring_b_t.unsqueeze(0).unsqueeze(2).expand(B, -1, 3), -f_total)
        
        # Ground collision
        below = pos[:, :, 1] < GROUND_Y  # (B, N)
        penetration = (GROUND_Y - pos[:, :, 1]).clamp(min=0)  # (B, N)
        forces[:, :, 1] += GROUND_K * penetration
        below_f = below.float().unsqueeze(2)
        forces[:, :, 0] -= FRICTION * vel[:, :, 0] * below_f.squeeze(2)
        forces[:, :, 2] -= FRICTION * vel[:, :, 2] * below_f.squeeze(2)
        
        # Drag
        forces -= DRAG * vel
        
        # External
        forces += ext_forces
        
        # Integration
        vel += forces * DT
        pos += vel * DT
    
    # Compute fitness
    end_com = pos.mean(dim=1)  # (B, 3)
    displacement_x = end_com[:, 0] - start_com_x  # (B,)
    
    # Z drift penalty
    drift_z = end_com[:, 2].abs()
    drift_penalty = drift_z * 1.0
    
    # Spread penalty
    extent = pos.max(dim=1).values - pos.min(dim=1).values  # (B, 3)
    spread_penalty = ((extent - 6.0).clamp(min=0) * 2.0).sum(dim=1)
    
    # Ground penalty
    below_ground = (pos[:, :, 1] < GROUND_Y - 1.0).float().sum(dim=1)
    ground_penalty = below_ground * 0.3
    
    # Energy penalty
    max_energy = N_PARTICLES * N_STEPS * (BASE_AMP * 1.5) ** 2 * 3
    norm_energy = total_energy / max_energy
    energy_penalty = 2.0 * norm_energy * 100
    
    fitness = displacement_x - drift_penalty - spread_penalty - ground_penalty - energy_penalty
    
    return fitness, displacement_x


# ===================================================
#  GA (on GPU tensors)
# ===================================================

def evolve_gpu():
    print("=" * 60)
    print("  GPU 3D Soft-Body Locomotion Evolution")
    print(f"  Body: {GRID_X}×{GRID_Y}×{GRID_Z} = {N_PARTICLES} particles")
    print(f"  NN: Input({INPUT_SIZE}) -> Hidden({HIDDEN_SIZE}) -> Output({OUTPUT_SIZE})")
    print(f"  Genome: {N_GENES} params | Pop: {BATCH_SIZE} (all on GPU!)")
    print(f"  Gen: {N_GENERATIONS}")
    print("=" * 60)
    
    # Build body template and move to GPU
    rest_pos, norm_pos, spring_a, spring_b, rest_lengths, n_springs = build_body_template()
    print(f"  Springs: {n_springs}")
    
    rest_pos_t = torch.tensor(rest_pos, dtype=torch.float32, device=DEVICE)
    norm_pos_t = torch.tensor(norm_pos, dtype=torch.float32, device=DEVICE)
    spring_a_t = torch.tensor(spring_a, dtype=torch.long, device=DEVICE)
    spring_b_t = torch.tensor(spring_b, dtype=torch.long, device=DEVICE)
    rest_lengths_t = torch.tensor(rest_lengths, dtype=torch.float32, device=DEVICE)
    
    # Initialize population on GPU
    scale_w1 = np.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE))
    scale_w2 = np.sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE))
    
    pop = torch.randn(BATCH_SIZE, N_GENES, device=DEVICE) * 0.3
    # Scale W1, W2 properly
    pop[:, :N_W1] *= scale_w1 / 0.3
    pop[:, N_W1:N_W1 + N_B1] = 0  # biases
    pop[:, N_W1 + N_B1:N_W1 + N_B1 + N_W2] *= scale_w2 / 0.3
    pop[:, N_W1 + N_B1 + N_W2:N_W1 + N_B1 + N_W2 + N_B2] = 0  # biases
    pop[:, -1] = torch.empty(BATCH_SIZE, device=DEVICE).uniform_(0.5, 3.0)  # freq
    
    pop_fitness = torch.full((BATCH_SIZE,), float('-inf'), device=DEVICE)
    
    history = {"best": [], "avg": []}
    n_elite = max(1, int(BATCH_SIZE * ELITE_FRAC))
    stagnation = 0
    prev_best = -float('inf')
    
    t0 = time.time()
    
    for gen in range(N_GENERATIONS):
        gen_t0 = time.time()
        
        # Find unevaluated
        unevaluated = pop_fitness == float('-inf')
        if unevaluated.any():
            mask = unevaluated
            fitness_new, _ = simulate_batch(
                pop[mask], rest_pos_t, norm_pos_t, spring_a_t, spring_b_t, rest_lengths_t
            )
            pop_fitness[mask] = fitness_new
        
        # Sort
        sorted_idx = pop_fitness.argsort(descending=True)
        pop = pop[sorted_idx]
        pop_fitness = pop_fitness[sorted_idx]
        
        best_fit = pop_fitness[0].item()
        avg_fit = pop_fitness.mean().item()
        history["best"].append(best_fit)
        history["avg"].append(avg_fit)
        
        if best_fit > prev_best + 0.1:
            stagnation = 0
            prev_best = best_fit
        else:
            stagnation += 1
        
        gen_dt = time.time() - gen_t0
        if gen % 10 == 0 or gen == N_GENERATIONS - 1:
            freq = abs(pop[0, -1].item())
            print(f"  Gen {gen:3d}/{N_GENERATIONS}: best={best_fit:+8.2f}  "
                  f"avg={avg_fit:+8.2f}  freq={freq:.2f}Hz  "
                  f"stag={stagnation}  [{gen_dt:.1f}s]")
        
        # Adaptive mutation
        sigma = MUTATION_SIGMA
        rate = MUTATION_RATE
        if stagnation > 20:
            sigma *= 1.5; rate *= 1.5
        if stagnation > 50:
            sigma *= 2.0; rate *= 2.0
        
        # Build next gen
        # Elites
        new_pop = pop[:n_elite].clone()
        new_fitness = pop_fitness[:n_elite].clone()
        
        # Fresh injection if stagnating
        n_fresh = 0
        if stagnation > 30:
            n_fresh = int(BATCH_SIZE * 0.1)
            fresh = torch.randn(n_fresh, N_GENES, device=DEVICE) * 0.3
            fresh[:, :N_W1] *= scale_w1 / 0.3
            fresh[:, N_W1:N_W1 + N_B1] = 0
            fresh[:, N_W1 + N_B1:N_W1 + N_B1 + N_W2] *= scale_w2 / 0.3
            fresh[:, N_W1 + N_B1 + N_W2:N_W1 + N_B1 + N_W2 + N_B2] = 0
            fresh[:, -1] = torch.empty(n_fresh, device=DEVICE).uniform_(0.5, 3.0)
            new_pop = torch.cat([new_pop, fresh])
            new_fitness = torch.cat([new_fitness,
                                      torch.full((n_fresh,), float('-inf'), device=DEVICE)])
        
        # Children via crossover + mutation
        n_children = BATCH_SIZE - new_pop.shape[0]
        
        # Tournament selection (on GPU)
        t_idx = torch.randint(BATCH_SIZE, (n_children, TOURNAMENT_SIZE), device=DEVICE)
        t_fitness = pop_fitness[t_idx]
        p1_idx = t_idx[torch.arange(n_children, device=DEVICE), t_fitness.argmax(dim=1)]
        
        t_idx2 = torch.randint(BATCH_SIZE, (n_children, TOURNAMENT_SIZE), device=DEVICE)
        t_fitness2 = pop_fitness[t_idx2]
        p2_idx = t_idx2[torch.arange(n_children, device=DEVICE), t_fitness2.argmax(dim=1)]
        
        # Uniform crossover
        parents1 = pop[p1_idx]
        parents2 = pop[p2_idx]
        mask = torch.rand(n_children, N_GENES, device=DEVICE) < 0.5
        children = torch.where(mask, parents1, parents2)
        
        # Mutation
        mut_mask = torch.rand(n_children, N_GENES, device=DEVICE) < rate
        mutations = torch.randn(n_children, N_GENES, device=DEVICE) * sigma
        mutations[:, -1] *= 0.3  # freq mutation gentler
        children += mutations * mut_mask.float()
        
        new_pop = torch.cat([new_pop, children])
        new_fitness = torch.cat([new_fitness,
                                  torch.full((n_children,), float('-inf'), device=DEVICE)])
        
        pop = new_pop
        pop_fitness = new_fitness
    
    total_time = time.time() - t0
    
    best_genes = pop[0].cpu().numpy()
    best_fitness = pop_fitness[0].item()
    print(f"\n  Total: {total_time / 60:.1f} min")
    print(f"  Best fitness: {best_fitness:+.2f}")
    
    return best_genes, best_fitness, history, total_time


# ===================================================
#  3D VISUALIZATION (same as CPU version)
# ===================================================

def visualize_3d(best_genes, best_fit, history, total_time):
    rest_pos, norm_pos, spring_a, spring_b, rest_lengths, n_springs = build_body_template()
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"GPU 3D Soft-Body Locomotion: {GRID_X}×{GRID_Y}×{GRID_Z} = {N_PARTICLES} particles\n"
                 f"NN({INPUT_SIZE}→{HIDDEN_SIZE}→{OUTPUT_SIZE}), "
                 f"{N_GENES} params | Best={best_fit:.2f} | {total_time / 60:.1f} min | "
                 f"Device: {DEVICE}",
                 fontsize=12, fontweight='bold')
    
    # Panel 1: Fitness
    ax1 = fig.add_subplot(2, 2, 1)
    gens = range(len(history["best"]))
    ax1.fill_between(gens, history["avg"], history["best"], alpha=0.15, color='#e74c3c')
    ax1.plot(gens, history["best"], '#e74c3c', linewidth=2, label='Best')
    ax1.plot(gens, history["avg"], '#f39c12', linewidth=1.5, alpha=0.6, label='Average')
    ax1.set_xlabel("Generation"); ax1.set_ylabel("Fitness")
    ax1.set_title("GPU 3D Fitness Evolution", fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    
    # Panel 2: Start config
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    springs_list = list(zip(spring_a, spring_b))
    lines = [[rest_pos[a], rest_pos[b]] for a, b in springs_list]
    lc = Line3DCollection(lines, colors='#3498db', alpha=0.3, linewidths=0.5)
    ax2.add_collection3d(lc)
    ax2.scatter(rest_pos[:, 0], rest_pos[:, 2], rest_pos[:, 1],
                c=rest_pos[:, 1], cmap='RdYlBu_r', s=80,
                edgecolors='black', linewidth=0.5)
    xx, zz = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-1.5, 1.5, 5))
    ax2.plot_surface(xx, zz, np.full_like(xx, GROUND_Y), alpha=0.15, color='green')
    ax2.set_xlabel('X'); ax2.set_ylabel('Z'); ax2.set_zlabel('Y')
    ax2.set_title("Initial Config", fontweight='bold')
    
    # Panel 3: After simulation (CPU replay of best)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    
    # Replay best genome on CPU for visualization
    rest_pos_t = torch.tensor(rest_pos, dtype=torch.float32, device=DEVICE)
    norm_pos_t = torch.tensor(norm_pos, dtype=torch.float32, device=DEVICE)
    spring_a_t = torch.tensor(spring_a, dtype=torch.long, device=DEVICE)
    spring_b_t = torch.tensor(spring_b, dtype=torch.long, device=DEVICE)
    rest_lengths_t = torch.tensor(rest_lengths, dtype=torch.float32, device=DEVICE)
    
    genes_t = torch.tensor(best_genes, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    
    # Run single sim batch=1
    B = 1
    pos = rest_pos_t.unsqueeze(0).clone()
    vel = torch.zeros(B, N_PARTICLES, 3, device=DEVICE)
    
    idx = 0
    W1 = genes_t[:, idx:idx + N_W1].reshape(B, INPUT_SIZE, HIDDEN_SIZE); idx += N_W1
    b1 = genes_t[:, idx:idx + N_B1].unsqueeze(1); idx += N_B1
    W2 = genes_t[:, idx:idx + N_W2].reshape(B, HIDDEN_SIZE, OUTPUT_SIZE); idx += N_W2
    b2 = genes_t[:, idx:idx + N_B2].unsqueeze(1); idx += N_B2
    freq_val = genes_t[:, idx].abs()
    norm_inp = norm_pos_t.unsqueeze(0)
    
    traj_x, traj_y, traj_z = [], [], []
    with torch.no_grad():
        for step in range(N_STEPS):
            t = step * DT
            sin_t = torch.sin(2 * np.pi * freq_val * t).unsqueeze(1).unsqueeze(2)
            cos_t = torch.cos(2 * np.pi * freq_val * t).unsqueeze(1).unsqueeze(2)
            nn_input = torch.cat([sin_t.expand(B, N_PARTICLES, 1),
                                   cos_t.expand(B, N_PARTICLES, 1), norm_inp], dim=2)
            hidden = torch.tanh(torch.bmm(nn_input, W1) + b1)
            output = torch.tanh(torch.bmm(hidden, W2) + b2)
            on_ground = (pos[:, :, 1] < GROUND_Y + 0.3).float()
            gc = 0.5 + 1.0 * on_ground
            ext = torch.zeros(B, N_PARTICLES, 3, device=DEVICE)
            ext[:, :, 0] = BASE_AMP * output[:, :, 0] * gc
            ext[:, :, 1] = BASE_AMP * torch.clamp(output[:, :, 1], min=0) * gc
            ext[:, :, 2] = BASE_AMP * output[:, :, 2] * gc * 0.5
            
            forces = torch.zeros(B, N_PARTICLES, 3, device=DEVICE)
            forces[:, :, 1] += GRAVITY
            pa = pos[:, spring_a_t]; pb = pos[:, spring_b_t]
            diff = pb - pa
            dist = torch.norm(diff, dim=2, keepdim=True).clamp(min=1e-8)
            direction = diff / dist
            stretch = dist - rest_lengths_t.unsqueeze(0).unsqueeze(2)
            rel_v = vel[:, spring_b_t] - vel[:, spring_a_t]
            vel_along = (rel_v * direction).sum(dim=2, keepdim=True)
            f_s = SPRING_K * stretch * direction
            f_d = SPRING_DAMP * vel_along * direction
            ft = f_s + f_d
            forces.scatter_add_(1, spring_a_t.unsqueeze(0).unsqueeze(2).expand(B, -1, 3), ft)
            forces.scatter_add_(1, spring_b_t.unsqueeze(0).unsqueeze(2).expand(B, -1, 3), -ft)
            pen = (GROUND_Y - pos[:, :, 1]).clamp(min=0)
            forces[:, :, 1] += GROUND_K * pen
            bl = (pos[:, :, 1] < GROUND_Y).float()
            forces[:, :, 0] -= FRICTION * vel[:, :, 0] * bl
            forces[:, :, 2] -= FRICTION * vel[:, :, 2] * bl
            forces -= DRAG * vel
            forces += ext
            vel += forces * DT
            pos += vel * DT
            
            if step % 25 == 0:
                com = pos[0].mean(dim=0).cpu().numpy()
                traj_x.append(com[0]); traj_y.append(com[1]); traj_z.append(com[2])
    
    pos_end = pos[0].cpu().numpy()
    end_com = pos_end.mean(axis=0)
    disp = end_com[0]
    
    lines_end = [[pos_end[a], pos_end[b]] for a, b in springs_list]
    lc_end = Line3DCollection(lines_end, colors='#e74c3c', alpha=0.3, linewidths=0.5)
    ax3.add_collection3d(lc_end)
    ax3.scatter(pos_end[:, 0], pos_end[:, 2], pos_end[:, 1],
                c=pos_end[:, 1], cmap='RdYlBu_r', s=80,
                edgecolors='black', linewidth=0.5)
    ax3.plot(traj_x, traj_z, traj_y, 'k--', linewidth=2, alpha=0.6, label='COM path')
    x_max = max(pos_end[:, 0].max() + 2, 5)
    xx, zz = np.meshgrid(np.linspace(-2, x_max, 5), np.linspace(-3, 3, 5))
    ax3.plot_surface(xx, zz, np.full_like(xx, GROUND_Y), alpha=0.1, color='green')
    ax3.set_xlabel('X'); ax3.set_ylabel('Z'); ax3.set_zlabel('Y')
    ax3.set_title(f"After {N_STEPS * DT:.1f}s: X={disp:.1f}", fontweight='bold')
    ax3.legend(fontsize=8)
    
    # Panel 4: Comparison bar
    ax4 = fig.add_subplot(2, 2, 4)
    methods = {"3D GPU\n(this)": disp, "2D NN\n(v1)": 261.65,
               "2D Optimal": 68.79, "2D CPG 12D": 41.35, "2D CPG 5D": 35.35}
    names = list(methods.keys())
    vals = list(methods.values())
    colors_bar = ['#e74c3c', '#95a5a6', '#f1c40f', '#2ecc71', '#3498db']
    bars = ax4.bar(range(len(names)), vals, color=colors_bar, alpha=0.85,
                  edgecolor='white', linewidth=2)
    for b, v in zip(bars, vals):
        ax4.text(b.get_x() + b.get_width() / 2, max(v, 0) + 2, f'{v:.1f}',
                ha='center', fontsize=10, fontweight='bold')
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, fontsize=9)
    ax4.set_ylabel("Forward Displacement")
    ax4.set_title("3D GPU vs 2D Methods", fontweight='bold')
    ax4.grid(True, alpha=0.2, axis='y')
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "evolution_3d_gpu.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path, disp


# ===================================================
#  MAIN
# ===================================================

def main():
    best_genes, best_fit, history, total_time = evolve_gpu()
    
    freq = abs(best_genes[-1])
    body_info = build_body_template()
    n_springs = body_info[5]
    
    print(f"\n{'=' * 60}")
    print(f"  GPU 3D LOCOMOTION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Body: {GRID_X}×{GRID_Y}×{GRID_Z} = {N_PARTICLES} particles, {n_springs} springs")
    print(f"  NN: {INPUT_SIZE}→{HIDDEN_SIZE}→{OUTPUT_SIZE} ({N_GENES} params)")
    print(f"  Freq: {freq:.3f} Hz")
    print(f"  Best fitness: {best_fit:.2f}")
    print(f"  Device: {DEVICE}")
    print(f"{'=' * 60}")
    
    fig_path, displacement = visualize_3d(best_genes, best_fit, history, total_time)
    
    results = {
        "experiment": "GPU 3D Soft-Body Locomotion Evolution",
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "N/A",
        "body": {"grid": f"{GRID_X}x{GRID_Y}x{GRID_Z}", "n_particles": N_PARTICLES,
                 "n_springs": n_springs},
        "nn": {"input": INPUT_SIZE, "hidden": HIDDEN_SIZE, "output": OUTPUT_SIZE,
               "total_params": N_GENES},
        "ga": {"population": BATCH_SIZE, "generations": N_GENERATIONS},
        "results": {"best_fitness": float(best_fit), "displacement_x": float(displacement),
                     "freq": float(freq)},
        "elapsed_min": round(total_time / 60, 1),
        "figure": fig_path,
    }
    
    log_path = os.path.join(RESULTS_DIR, "evolution_3d_gpu_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Log: {log_path}")
    
    # Beep!
    try:
        import winsound
        for _ in range(5):
            winsound.Beep(1200, 200)
            time.sleep(0.15)
        print("\n  🔔 BEEP! GPU 3D experiment complete!")
    except Exception:
        print("\n  Done!")


if __name__ == "__main__":
    main()
