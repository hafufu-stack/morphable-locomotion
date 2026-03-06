"""
3D Soft-Body Locomotion Evolution
==================================

Extends the 2D morphable-locomotion to full 3D physics:
  - 3D particle grid (6×3×3 = 54 particles)
  - 3D spring forces (Delaunay tetrahedralization)
  - 3D ground collision (XZ plane)
  - NN Controller: Input(5) -> Hidden(24, tanh) -> Output(3)
    Input:  [sin(2π·f·t), cos(2π·f·t), norm_x, norm_y, norm_z]
    Output: [Fx, Fy, Fz] per particle

CPU 22-core parallel GA evolution + 3D visualization.

Usage:
    python src/evolve_3d.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import Delaunay
import os, time, json
from multiprocessing import Pool, cpu_count

N_WORKERS = max(1, cpu_count() - 2)

# === Physics Config ===
GRID_X, GRID_Y, GRID_Z = 6, 3, 3  # Body shape: long, low, medium-width
N_PARTICLES = GRID_X * GRID_Y * GRID_Z  # 54 particles
SPACING = 0.5
DT = 0.012       # Slightly smaller for 3D stability
N_STEPS = 500    # More steps for 3D (7.5 seconds → longer to observe gait)
SPRING_K = 25.0  # Stiffer for 3D stability
SPRING_DAMP = 1.2
DRAG = 0.4
GROUND_Y = -0.5
GROUND_K = 600.0
GRAVITY = -9.8   # More realistic gravity
FRICTION = 3.0   # Ground friction coefficient
BASE_AMP = 35.0  # Force amplitude

# === GA Config ===
POP_SIZE = 300
N_GENERATIONS = 200
MUTATION_RATE = 0.15
MUTATION_SIGMA = 0.3
ELITE_FRAC = 0.05
TOURNAMENT_SIZE = 5

# === NN Config ===
HIDDEN_SIZE = 24   # Slightly larger for 3D
INPUT_SIZE = 5     # [sin, cos, nx, ny, nz]
OUTPUT_SIZE = 3    # [Fx, Fy, Fz]

# Genome layout
N_W1 = INPUT_SIZE * HIDDEN_SIZE   # 5*24 = 120
N_B1 = HIDDEN_SIZE                # 24
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE  # 24*3 = 72
N_B2 = OUTPUT_SIZE                # 3
N_FREQ = 1                        # 1
N_GENES = N_W1 + N_B1 + N_W2 + N_B2 + N_FREQ  # 220

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================================================
#  3D SOFT BODY
# ===================================================

class SoftBody3D:
    """3D deformable body: a grid of particles connected by springs."""
    
    def __init__(self, cx=0, cy=2.0, cz=0):
        self.n = N_PARTICLES
        self.pos = np.zeros((self.n, 3))
        self.vel = np.zeros((self.n, 3))
        self.mass = np.ones(self.n)
        
        # Create 3D grid
        idx = 0
        for gx in range(GRID_X):
            for gy in range(GRID_Y):
                for gz in range(GRID_Z):
                    self.pos[idx] = [
                        cx + gx * SPACING - (GRID_X - 1) * SPACING / 2,
                        cy + gy * SPACING,
                        cz + gz * SPACING - (GRID_Z - 1) * SPACING / 2,
                    ]
                    idx += 1
        
        self.rest_pos = self.pos.copy()
        
        # Build springs via Delaunay tetrahedralization
        tri = Delaunay(self.pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        self.springs = list(edges)
        self.rest_lengths = {}
        for a, b in self.springs:
            self.rest_lengths[(a, b)] = np.linalg.norm(self.pos[a] - self.pos[b])
        
        # Precompute normalized rest positions for NN input
        self.norm_pos = np.zeros((self.n, 3))
        for dim in range(3):
            vmin = self.rest_pos[:, dim].min()
            vmax = self.rest_pos[:, dim].max()
            rng = vmax - vmin + 1e-8
            self.norm_pos[:, dim] = 2 * (self.rest_pos[:, dim] - vmin) / rng - 1
    
    def step(self, external_forces):
        forces = np.zeros_like(self.pos)
        
        # Gravity (Y-axis)
        forces[:, 1] += GRAVITY * self.mass
        
        # Spring forces
        for a, b in self.springs:
            diff = self.pos[b] - self.pos[a]
            dist = np.linalg.norm(diff) + 1e-8
            direction = diff / dist
            rest = self.rest_lengths[(a, b)]
            stretch = dist - rest
            rel_vel = self.vel[b] - self.vel[a]
            f_spring = SPRING_K * stretch * direction
            f_damp = SPRING_DAMP * np.dot(rel_vel, direction) * direction
            forces[a] += f_spring + f_damp
            forces[b] -= f_spring + f_damp
        
        # Ground collision (Y = GROUND_Y plane)
        for i in range(self.n):
            if self.pos[i, 1] < GROUND_Y:
                penetration = GROUND_Y - self.pos[i, 1]
                forces[i, 1] += GROUND_K * penetration
                # Friction: resist XZ sliding when on ground
                forces[i, 0] -= FRICTION * self.vel[i, 0]
                forces[i, 2] -= FRICTION * self.vel[i, 2]
        
        # Air drag
        forces -= DRAG * self.vel
        
        # External forces
        forces += external_forces
        
        # Integration (Euler)
        acc = forces / self.mass[:, np.newaxis]
        self.vel += acc * DT
        self.pos += self.vel * DT
    
    def center_of_mass(self):
        return np.mean(self.pos, axis=0)


# ===================================================
#  NN CONTROLLER
# ===================================================

def nn_unpack(genes):
    """Unpack genome into NN weights."""
    idx = 0
    W1 = genes[idx:idx + N_W1].reshape(INPUT_SIZE, HIDDEN_SIZE)
    idx += N_W1
    b1 = genes[idx:idx + N_B1]
    idx += N_B1
    W2 = genes[idx:idx + N_W2].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    idx += N_W2
    b2 = genes[idx:idx + N_B2]
    idx += N_B2
    freq = abs(genes[idx])
    return W1, b1, W2, b2, freq


def nn_generate_forces(genes, body, t):
    """Generate NN-controlled 3D forces for all particles."""
    n = body.n
    forces = np.zeros((n, 3))
    W1, b1, W2, b2, freq = nn_unpack(genes)
    
    sin_t = np.sin(2 * np.pi * freq * t)
    cos_t = np.cos(2 * np.pi * freq * t)
    
    for i in range(n):
        inp = np.array([sin_t, cos_t,
                        body.norm_pos[i, 0],
                        body.norm_pos[i, 1],
                        body.norm_pos[i, 2]])
        
        hidden = np.tanh(inp @ W1 + b1)
        output = np.tanh(hidden @ W2 + b2)
        
        # Ground contact modulation
        ground_contact = 1.5 if body.pos[i, 1] < GROUND_Y + 0.3 else 0.5
        
        forces[i, 0] = BASE_AMP * output[0] * ground_contact  # Fx: forward
        forces[i, 1] = BASE_AMP * max(0, output[1]) * ground_contact  # Fy: up only
        forces[i, 2] = BASE_AMP * output[2] * ground_contact * 0.5  # Fz: lateral (weaker)
    
    return forces


# ===================================================
#  FITNESS with Energy Penalty
# ===================================================

def evaluate_fitness_3d(genes_array):
    """Evaluate 3D NN genome fitness."""
    body = SoftBody3D(cx=0, cy=2.0, cz=0)
    start_com = body.center_of_mass()
    
    total_energy = 0.0
    
    for step in range(N_STEPS):
        t = step * DT
        forces = nn_generate_forces(genes_array, body, t)
        body.step(forces)
        total_energy += np.sum(forces ** 2)
    
    end_com = body.center_of_mass()
    
    # Primary: forward displacement (X direction)
    displacement_x = end_com[0] - start_com[0]
    
    # Penalty: lateral drift (Z direction)
    drift_z = abs(end_com[2] - start_com[2])
    drift_penalty = drift_z * 1.0
    
    # Penalty: body disintegration
    extent = np.max(body.pos, axis=0) - np.min(body.pos, axis=0)
    spread_penalty = (max(0, extent[0] - 6.0) * 2.0 +
                      max(0, extent[1] - 6.0) * 2.0 +
                      max(0, extent[2] - 6.0) * 2.0)
    
    # Penalty: particles below ground
    below_ground = np.sum(body.pos[:, 1] < GROUND_Y - 1.0)
    ground_penalty = below_ground * 0.3
    
    # Energy penalty (moderate)
    max_energy = N_PARTICLES * N_STEPS * (BASE_AMP * 1.5) ** 2 * 3
    norm_energy = total_energy / max_energy
    energy_penalty = 2.0 * norm_energy * 100
    
    fitness = displacement_x - drift_penalty - spread_penalty - ground_penalty - energy_penalty
    
    return fitness


# ===================================================
#  GA
# ===================================================

def make_nn_genome():
    """Create random NN genome for 3D."""
    scale_w1 = np.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE))
    scale_w2 = np.sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE))
    w1 = np.random.randn(N_W1) * scale_w1
    b1 = np.zeros(N_B1)
    w2 = np.random.randn(N_W2) * scale_w2
    b2 = np.zeros(N_B2)
    freq = np.array([np.random.uniform(0.5, 3.0)])
    return np.concatenate([w1, b1, w2, b2, freq])


def mutate_genes(genes, rate=MUTATION_RATE, sigma=MUTATION_SIGMA):
    child = genes.copy()
    mask = np.random.random(N_GENES) < rate
    mutations = np.random.randn(N_GENES) * sigma
    mutations[-1] *= 0.3
    child[mask] += mutations[mask]
    return child


def crossover_genes(g1, g2):
    child = g1.copy()
    if np.random.random() < 0.3:
        boundaries = [0, N_W1, N_W1 + N_B1, N_W1 + N_B1 + N_W2,
                      N_W1 + N_B1 + N_W2 + N_B2, N_GENES]
        for s in range(len(boundaries) - 1):
            if np.random.random() < 0.5:
                child[boundaries[s]:boundaries[s + 1]] = g2[boundaries[s]:boundaries[s + 1]]
    else:
        mask = np.random.random(N_GENES) < 0.5
        child[mask] = g2[mask]
    return child


def evolve():
    print("=" * 60)
    print("  3D Soft-Body Locomotion Evolution")
    print(f"  Body: {GRID_X}×{GRID_Y}×{GRID_Z} = {N_PARTICLES} particles")
    print(f"  Springs: built via 3D Delaunay")
    print(f"  NN: Input({INPUT_SIZE}) -> Hidden({HIDDEN_SIZE}) -> Output({OUTPUT_SIZE})")
    print(f"  Genome: {N_GENES} parameters")
    print(f"  Pop={POP_SIZE}, Gen={N_GENERATIONS}")
    print(f"  Workers: {N_WORKERS} cores")
    print("=" * 60)
    
    t0 = time.time()
    pop_genes = [make_nn_genome() for _ in range(POP_SIZE)]
    pop_fitness = [-np.inf] * POP_SIZE
    
    history = {"best": [], "avg": []}
    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))
    stagnation = 0
    prev_best = -np.inf
    
    for gen in range(N_GENERATIONS):
        gen_t0 = time.time()
        
        to_eval = [(i, pop_genes[i]) for i in range(POP_SIZE) if pop_fitness[i] == -np.inf]
        if to_eval:
            indices = [x[0] for x in to_eval]
            genes_list = [x[1].copy() for x in to_eval]
            with Pool(N_WORKERS) as pool:
                fits = pool.map(evaluate_fitness_3d, genes_list)
            for idx, fit in zip(indices, fits):
                pop_fitness[idx] = fit
        
        order = np.argsort(pop_fitness)[::-1]
        pop_genes = [pop_genes[i] for i in order]
        pop_fitness = [pop_fitness[i] for i in order]
        
        best_fit = pop_fitness[0]
        avg_fit = np.mean(pop_fitness)
        history["best"].append(best_fit)
        history["avg"].append(avg_fit)
        
        if best_fit > prev_best + 0.1:
            stagnation = 0
            prev_best = best_fit
        else:
            stagnation += 1
        
        gen_dt = time.time() - gen_t0
        if gen % 10 == 0 or gen == N_GENERATIONS - 1:
            freq = abs(pop_genes[0][-1])
            print(f"  Gen {gen:3d}/{N_GENERATIONS}: best={best_fit:+8.2f}  "
                  f"avg={avg_fit:+8.2f}  freq={freq:.2f}Hz  "
                  f"stag={stagnation}  [{gen_dt:.1f}s]")
        
        sigma = MUTATION_SIGMA
        rate = MUTATION_RATE
        if stagnation > 20:
            sigma *= 1.5; rate *= 1.5
        if stagnation > 50:
            sigma *= 2.0; rate *= 2.0
        
        new_genes = [pop_genes[i].copy() for i in range(n_elite)]
        new_fitness = [pop_fitness[i] for i in range(n_elite)]
        
        if stagnation > 30:
            for _ in range(int(POP_SIZE * 0.1)):
                new_genes.append(make_nn_genome())
                new_fitness.append(-np.inf)
        
        while len(new_genes) < POP_SIZE:
            c1 = np.random.choice(len(pop_genes), TOURNAMENT_SIZE, replace=False)
            p1 = pop_genes[max(c1, key=lambda i: pop_fitness[i])]
            c2 = np.random.choice(len(pop_genes), TOURNAMENT_SIZE, replace=False)
            p2 = pop_genes[max(c2, key=lambda i: pop_fitness[i])]
            child = crossover_genes(p1, p2)
            child = mutate_genes(child, rate=rate, sigma=sigma)
            new_genes.append(child)
            new_fitness.append(-np.inf)
        
        pop_genes = new_genes
        pop_fitness = new_fitness
    
    total_time = time.time() - t0
    print(f"\n  Total: {total_time / 60:.1f} min, Best: {pop_fitness[0]:+.2f}")
    
    return pop_genes[0], pop_fitness[0], history, total_time


# ===================================================
#  3D VISUALIZATION
# ===================================================

def visualize_3d(best_genes, best_fit, history, total_time):
    """Create multi-panel 3D visualization."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"3D Soft-Body Locomotion: {GRID_X}×{GRID_Y}×{GRID_Z} = {N_PARTICLES} particles\n"
                 f"NN({INPUT_SIZE}→{HIDDEN_SIZE}→{OUTPUT_SIZE}), "
                 f"{N_GENES} params | Best={best_fit:.2f} | {total_time / 60:.1f} min",
                 fontsize=13, fontweight='bold')
    
    # === Panel 1: Fitness Evolution ===
    ax1 = fig.add_subplot(2, 2, 1)
    gens = range(len(history["best"]))
    ax1.plot(gens, history["best"], '#e74c3c', linewidth=2, label='Best')
    ax1.plot(gens, history["avg"], '#f39c12', linewidth=1.5, alpha=0.6, label='Average')
    ax1.fill_between(gens, history["avg"], history["best"], alpha=0.15, color='#e74c3c')
    ax1.set_xlabel("Generation"); ax1.set_ylabel("Fitness")
    ax1.set_title("3D Fitness Evolution", fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    
    # === Panel 2: 3D Body at Start ===
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    body_start = SoftBody3D()
    pos = body_start.pos
    
    # Draw springs
    lines = []
    for a, b in body_start.springs:
        lines.append([pos[a], pos[b]])
    lc = Line3DCollection(lines, colors='#3498db', alpha=0.3, linewidths=0.5)
    ax2.add_collection3d(lc)
    
    # Draw particles colored by height
    sc = ax2.scatter(pos[:, 0], pos[:, 2], pos[:, 1],
                     c=pos[:, 1], cmap='RdYlBu_r', s=80,
                     edgecolors='black', linewidth=0.5, depthshade=True)
    
    # Ground plane
    xx, zz = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-1.5, 1.5, 5))
    yy = np.full_like(xx, GROUND_Y)
    ax2.plot_surface(xx, zz, yy, alpha=0.15, color='green')
    
    ax2.set_xlabel('X (forward)'); ax2.set_ylabel('Z (lateral)'); ax2.set_zlabel('Y (up)')
    ax2.set_title("Initial Body Configuration", fontweight='bold')
    
    # === Panel 3: 3D Body After Simulation (best genome) ===
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    body_end = SoftBody3D()
    trajectory_x = []
    trajectory_y = []
    trajectory_z = []
    
    for step in range(N_STEPS):
        t = step * DT
        forces = nn_generate_forces(best_genes, body_end, t)
        body_end.step(forces)
        if step % 20 == 0:
            com = body_end.center_of_mass()
            trajectory_x.append(com[0])
            trajectory_y.append(com[1])
            trajectory_z.append(com[2])
    
    pos_end = body_end.pos
    end_com = body_end.center_of_mass()
    
    # Draw springs
    lines_end = []
    for a, b in body_end.springs:
        lines_end.append([pos_end[a], pos_end[b]])
    lc_end = Line3DCollection(lines_end, colors='#e74c3c', alpha=0.3, linewidths=0.5)
    ax3.add_collection3d(lc_end)
    
    # Draw particles
    ax3.scatter(pos_end[:, 0], pos_end[:, 2], pos_end[:, 1],
                c=pos_end[:, 1], cmap='RdYlBu_r', s=80,
                edgecolors='black', linewidth=0.5, depthshade=True)
    
    # COM trajectory
    ax3.plot(trajectory_x, trajectory_z, trajectory_y,
             'k--', linewidth=2, alpha=0.6, label='COM path')
    
    # Ground plane (extended)
    x_max = max(pos_end[:, 0].max() + 2, 5)
    xx, zz = np.meshgrid(np.linspace(-2, x_max, 5), np.linspace(-3, 3, 5))
    yy = np.full_like(xx, GROUND_Y)
    ax3.plot_surface(xx, zz, yy, alpha=0.1, color='green')
    
    disp = end_com[0] - 0  # started at x=0
    ax3.set_xlabel('X (forward)'); ax3.set_ylabel('Z (lateral)'); ax3.set_zlabel('Y (up)')
    ax3.set_title(f"After {N_STEPS*DT:.1f}s: Displaced {disp:.1f} units", fontweight='bold')
    ax3.legend(fontsize=8)
    
    # === Panel 4: Force Over Time ===
    ax4 = fig.add_subplot(2, 2, 4)
    body_viz = SoftBody3D()
    times = np.arange(N_STEPS) * DT
    particles_show = [0, N_PARTICLES // 4, N_PARTICLES // 2,
                      3 * N_PARTICLES // 4, N_PARTICLES - 1]
    colors_p = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#3498db']
    labels_p = [f'P{p}' for p in particles_show]
    
    for pidx, col, lab in zip(particles_show, colors_p, labels_p):
        fx_hist = []
        for step in range(N_STEPS):
            forces = nn_generate_forces(best_genes, body_viz, step * DT)
            fx_hist.append(forces[pidx, 0])
        ax4.plot(times, fx_hist, color=col, linewidth=1.0, label=lab, alpha=0.8)
    
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Fx Force")
    ax4.set_title("Force Patterns (X-direction)", fontweight='bold')
    ax4.legend(fontsize=7, ncol=2); ax4.grid(True, alpha=0.2)
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "evolution_3d.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path, disp


# ===================================================
#  MAIN
# ===================================================

def main():
    best_genes, best_fit, history, total_time = evolve()
    
    freq = abs(best_genes[-1])
    print(f"\n{'=' * 60}")
    print(f"  3D LOCOMOTION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  Body: {GRID_X}×{GRID_Y}×{GRID_Z} = {N_PARTICLES} particles")
    print(f"  Springs: {len(SoftBody3D().springs)}")
    print(f"  NN: {INPUT_SIZE} → {HIDDEN_SIZE} → {OUTPUT_SIZE} ({N_GENES} params)")
    print(f"  Frequency: {freq:.3f} Hz")
    print(f"  Best fitness: {best_fit:.2f}")
    print(f"{'=' * 60}")
    
    fig_path, displacement = visualize_3d(best_genes, best_fit, history, total_time)
    
    body_info = SoftBody3D()
    results = {
        "experiment": "3D Soft-Body Locomotion Evolution",
        "body": {
            "grid": f"{GRID_X}x{GRID_Y}x{GRID_Z}",
            "n_particles": N_PARTICLES,
            "n_springs": len(body_info.springs),
        },
        "nn": {
            "input": INPUT_SIZE,
            "hidden": HIDDEN_SIZE,
            "output": OUTPUT_SIZE,
            "total_params": N_GENES,
        },
        "ga": {
            "population": POP_SIZE,
            "generations": N_GENERATIONS,
        },
        "results": {
            "best_fitness": float(best_fit),
            "displacement_x": float(displacement),
            "evolved_freq": float(freq),
        },
        "elapsed_min": round(total_time / 60, 1),
        "figure": fig_path,
    }
    
    log_path = os.path.join(RESULTS_DIR, "evolution_3d_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Log: {log_path}")
    
    # Beep when done
    try:
        import winsound
        for _ in range(5):
            winsound.Beep(1000, 200)
            time.sleep(0.15)
        print("\n  🔔 BEEP! 3D experiment complete!")
    except Exception:
        print("\n  Done!")


if __name__ == "__main__":
    main()
