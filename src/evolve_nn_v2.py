"""
Neural Network Controller Evolution v2: Constrained Locomotion
===============================================================

v1 discovered that unconstrained NN learns a trivial "constant push"
strategy (all particles at max force → 261.65, 3.8× Optimal).

v2 adds two constraints to force genuine WALKING behavior:
  1. Energy penalty: fitness -= α * Σ|force|²  (penalize brute force)
  2. Gait reward: bonus for Y-COM oscillation (reward walking pattern)

This forces the NN to discover energy-efficient locomotion patterns
that resemble biological walking, not just sliding.

Tests multiple α values to find the sweet spot.

CPU-only, 22-core parallel. ~30 min estimated.

Usage:
    python src/evolve_nn_v2.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os, time, json
from multiprocessing import Pool, cpu_count

N_WORKERS = max(1, cpu_count() - 2)

# === Physics Config (same as all others) ===
N_PARTICLES = 36
GRID_SIZE = 6
DT = 0.015
N_STEPS = 400
SPRING_K = 20.0
SPRING_DAMP = 1.0
DRAG = 0.3
GROUND_Y = -0.5
GROUND_K = 500.0
GRAVITY = -8.0
BASE_AMP = 40.0

# === GA Config ===
POP_SIZE = 300
N_GENERATIONS = 300
MUTATION_RATE = 0.15
MUTATION_SIGMA = 0.3
ELITE_FRAC = 0.05
TOURNAMENT_SIZE = 5

# === NN Config ===
HIDDEN_SIZE = 16
INPUT_SIZE = 4     # [sin(2π·f·t), cos(2π·f·t), norm_x_i, norm_y_i]
OUTPUT_SIZE = 2    # [Fx, Fy]

# NN genome size
N_W1 = INPUT_SIZE * HIDDEN_SIZE   # 64
N_B1 = HIDDEN_SIZE                # 16
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE  # 32
N_B2 = OUTPUT_SIZE                # 2
N_FREQ = 1                        # 1
N_GENES = N_W1 + N_B1 + N_W2 + N_B2 + N_FREQ  # 115

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================================================
#  SOFT BODY (identical to all other scripts)
# ===================================================

class SoftBody:
    def __init__(self, cx=0, cy=2.0):
        self.n = N_PARTICLES
        self.pos = np.zeros((self.n, 2))
        self.vel = np.zeros((self.n, 2))
        self.mass = np.ones(self.n)
        idx = 0
        spacing = 0.5
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                self.pos[idx] = [cx + col * spacing - (GRID_SIZE-1)*spacing/2,
                                 cy + row * spacing]
                idx += 1
        self.rest_pos = self.pos.copy()
        tri = Delaunay(self.pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        self.springs = list(edges)
        self.rest_lengths = {}
        for a, b in self.springs:
            self.rest_lengths[(a, b)] = np.linalg.norm(self.pos[a] - self.pos[b])

    def step(self, external_forces):
        forces = np.zeros_like(self.pos)
        forces[:, 1] += GRAVITY * self.mass
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
        for i in range(self.n):
            if self.pos[i, 1] < GROUND_Y:
                forces[i, 1] += GROUND_K * (GROUND_Y - self.pos[i, 1])
                forces[i, 0] -= 3.0 * self.vel[i, 0]
        forces -= DRAG * self.vel
        forces += external_forces
        acc = forces / self.mass[:, np.newaxis]
        self.vel += acc * DT
        self.pos += self.vel * DT

    def center_of_mass(self):
        return np.mean(self.pos, axis=0)


# ===================================================
#  NN FORWARD PASS (shared across all conditions)
# ===================================================

def nn_unpack(genes):
    """Unpack genome into NN weights."""
    idx = 0
    W1 = genes[idx:idx+N_W1].reshape(INPUT_SIZE, HIDDEN_SIZE)
    idx += N_W1
    b1 = genes[idx:idx+N_B1]
    idx += N_B1
    W2 = genes[idx:idx+N_W2].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    idx += N_W2
    b2 = genes[idx:idx+N_B2]
    idx += N_B2
    freq = abs(genes[idx])
    return W1, b1, W2, b2, freq


def nn_generate_forces(genes, body, t):
    """Generate NN-controlled forces for all particles."""
    n = body.n
    forces = np.zeros((n, 2))
    W1, b1, W2, b2, freq = nn_unpack(genes)
    
    sin_t = np.sin(2 * np.pi * freq * t)
    cos_t = np.cos(2 * np.pi * freq * t)
    
    x_min = body.rest_pos[:, 0].min()
    x_max = body.rest_pos[:, 0].max()
    y_min = body.rest_pos[:, 1].min()
    y_max = body.rest_pos[:, 1].max()
    x_range = x_max - x_min + 1e-8
    y_range = y_max - y_min + 1e-8
    
    for i in range(n):
        nx = 2 * (body.rest_pos[i, 0] - x_min) / x_range - 1
        ny = 2 * (body.rest_pos[i, 1] - y_min) / y_range - 1
        inp = np.array([sin_t, cos_t, nx, ny])
        hidden = np.tanh(inp @ W1 + b1)
        output = np.tanh(hidden @ W2 + b2)
        ground_contact = 1.5 if body.pos[i, 1] < GROUND_Y + 0.3 else 0.5
        forces[i, 0] = BASE_AMP * output[0] * ground_contact
        forces[i, 1] = BASE_AMP * max(0, output[1]) * ground_contact
    
    return forces


# ===================================================
#  FITNESS with CONSTRAINTS
# ===================================================

def evaluate_with_constraints(genes_array, alpha_energy=0.0, beta_gait=0.0):
    """Evaluate NN with energy penalty and gait reward."""
    body = SoftBody(cx=0, cy=2.0)
    start_com = body.center_of_mass()
    
    total_energy = 0.0
    y_com_history = []
    
    for step in range(N_STEPS):
        t = step * DT
        forces = nn_generate_forces(genes_array, body, t)
        body.step(forces)
        
        # Track energy: sum of squared forces (normalized)
        total_energy += np.sum(forces ** 2)
        
        # Track Y-COM for gait detection
        if step % 10 == 0:  # Sample every 10 steps
            y_com_history.append(body.center_of_mass()[1])
    
    end_com = body.center_of_mass()
    displacement = end_com[0] - start_com[0]
    
    # Standard penalties (same as CPG)
    extent = np.max(body.pos, axis=0) - np.min(body.pos, axis=0)
    spread_penalty = max(0, extent[0] - 5.0) * 2.0 + max(0, extent[1] - 5.0) * 2.0
    below_ground = np.sum(body.pos[:, 1] < GROUND_Y - 1.0)
    ground_penalty = below_ground * 0.5
    
    # Energy penalty: normalize by max possible energy
    # Max energy = N_PARTICLES * N_STEPS * (BASE_AMP * 1.5)² * 2 (x+y)
    max_energy = N_PARTICLES * N_STEPS * (BASE_AMP * 1.5) ** 2 * 2
    norm_energy = total_energy / max_energy  # 0 to 1
    energy_penalty = alpha_energy * norm_energy * 100  # Scale to meaningful range
    
    # Gait reward: Y-COM oscillation variance
    # Walking should produce rhythmic up-down movement
    if len(y_com_history) > 2:
        y_com = np.array(y_com_history)
        # High-pass filter: remove trend, keep oscillation
        y_detrended = y_com - np.linspace(y_com[0], y_com[-1], len(y_com))
        gait_var = np.std(y_detrended)
        gait_reward = beta_gait * min(gait_var * 50, 10)  # Cap at 10
    else:
        gait_reward = 0.0
    
    fitness = displacement - spread_penalty - ground_penalty - energy_penalty + gait_reward
    
    return fitness, displacement, norm_energy, gait_reward


# Per-condition evaluation functions (top-level for multiprocessing)
def eval_condition_A(genes):
    """Moderate energy penalty (α=0.5) + gait reward (β=1.0)"""
    fit, disp, energy, gait = evaluate_with_constraints(genes, alpha_energy=0.5, beta_gait=1.0)
    return fit

def eval_condition_B(genes):
    """Strong energy penalty (α=2.0) + strong gait reward (β=2.0)"""
    fit, disp, energy, gait = evaluate_with_constraints(genes, alpha_energy=2.0, beta_gait=2.0)
    return fit

def eval_condition_C(genes):
    """Very strong energy penalty (α=5.0) + gait reward (β=3.0)"""
    fit, disp, energy, gait = evaluate_with_constraints(genes, alpha_energy=5.0, beta_gait=3.0)
    return fit


# ===================================================
#  GA (reusable for each condition)
# ===================================================

def make_nn_genome():
    """Create a random NN genome."""
    scale_w1 = np.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE))
    scale_w2 = np.sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE))
    w1 = np.random.randn(N_W1) * scale_w1
    b1 = np.zeros(N_B1)
    w2 = np.random.randn(N_W2) * scale_w2
    b2 = np.zeros(N_B2)
    freq = np.array([np.random.uniform(0.5, 3.0)])
    return np.concatenate([w1, b1, w2, b2, freq])


def mutate_genes(genes, rate=MUTATION_RATE, sigma=MUTATION_SIGMA):
    """Mutate a genome."""
    child = genes.copy()
    mask = np.random.random(N_GENES) < rate
    mutations = np.random.randn(N_GENES) * sigma
    mutations[-1] *= 0.3  # freq mutation is gentler
    child[mask] += mutations[mask]
    return child


def crossover_genes(g1, g2):
    """Crossover two genomes."""
    child = g1.copy()
    if np.random.random() < 0.3:
        # Segment crossover
        boundaries = [0, N_W1, N_W1+N_B1, N_W1+N_B1+N_W2, N_W1+N_B1+N_W2+N_B2, N_GENES]
        for s in range(len(boundaries)-1):
            if np.random.random() < 0.5:
                child[boundaries[s]:boundaries[s+1]] = g2[boundaries[s]:boundaries[s+1]]
    else:
        mask = np.random.random(N_GENES) < 0.5
        child[mask] = g2[mask]
    return child


def evolve_condition(eval_func, condition_name, n_gens=N_GENERATIONS):
    """Run GA for one condition."""
    print(f"\n  {'='*50}")
    print(f"  CONDITION: {condition_name}")
    print(f"  Pop={POP_SIZE}, Gen={n_gens}, Genome={N_GENES}")
    print(f"  {'='*50}")
    
    t0 = time.time()
    
    # Initialize population
    pop_genes = [make_nn_genome() for _ in range(POP_SIZE)]
    pop_fitness = [-np.inf] * POP_SIZE
    
    history = {"best": [], "avg": []}
    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))
    stagnation = 0
    prev_best = -np.inf
    
    for gen in range(n_gens):
        gen_t0 = time.time()
        
        # Find unevaluated
        to_eval = [(i, pop_genes[i]) for i in range(POP_SIZE) if pop_fitness[i] == -np.inf]
        if to_eval:
            indices = [x[0] for x in to_eval]
            genes_list = [x[1].copy() for x in to_eval]
            with Pool(N_WORKERS) as pool:
                fits = pool.map(eval_func, genes_list)
            for idx, fit in zip(indices, fits):
                pop_fitness[idx] = fit
        
        # Sort by fitness
        order = np.argsort(pop_fitness)[::-1]
        pop_genes = [pop_genes[i] for i in order]
        pop_fitness = [pop_fitness[i] for i in order]
        
        best_fit = pop_fitness[0]
        avg_fit = np.mean(pop_fitness)
        history["best"].append(best_fit)
        history["avg"].append(avg_fit)
        
        # Stagnation
        if best_fit > prev_best + 0.1:
            stagnation = 0
            prev_best = best_fit
        else:
            stagnation += 1
        
        gen_dt = time.time() - gen_t0
        if gen % 20 == 0 or gen == n_gens - 1:
            freq = abs(pop_genes[0][-1])
            marker = " 🏆" if best_fit > 68.79 else ""
            print(f"    Gen {gen:3d}: best={best_fit:+8.2f}  avg={avg_fit:+8.2f}  "
                  f"freq={freq:.2f}Hz  stag={stagnation}  [{gen_dt:.1f}s]{marker}")
        
        # Adaptive mutation
        sigma = MUTATION_SIGMA
        rate = MUTATION_RATE
        if stagnation > 20:
            sigma *= 1.5
            rate *= 1.5
        if stagnation > 50:
            sigma *= 2.0
            rate *= 2.0
        
        # Build next gen
        new_genes = [pop_genes[i].copy() for i in range(n_elite)]
        new_fitness = [pop_fitness[i] for i in range(n_elite)]
        
        # Inject fresh if stagnating
        if stagnation > 30:
            n_fresh = int(POP_SIZE * 0.1)
            for _ in range(n_fresh):
                new_genes.append(make_nn_genome())
                new_fitness.append(-np.inf)
        
        while len(new_genes) < POP_SIZE:
            # Tournament select
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
    
    # Get best and its detailed stats
    best_genes = pop_genes[0]
    best_fit = pop_fitness[0]
    _, disp, energy, gait = evaluate_with_constraints(best_genes, 
                                                       alpha_energy=0.5, beta_gait=1.0)
    
    print(f"    → Finished in {total_time/60:.1f} min. Best={best_fit:+.2f}, Disp={disp:.2f}")
    
    return best_genes, best_fit, history, total_time


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize_all(results_dict):
    """Create comparison figure for all conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle("NN Controller v2: Energy-Constrained Locomotion\n"
                 "Forcing genuine walking patterns instead of brute-force sliding",
                 fontsize=13, fontweight="bold")
    
    # Panel 1: Fitness evolution for all conditions
    ax = axes[0][0]
    colors = {'A': '#e74c3c', 'B': '#3498db', 'C': '#2ecc71'}
    for key, data in results_dict.items():
        label = data['label']
        hist = data['history']
        ax.plot(hist['best'], color=colors[key], linewidth=2, label=f"{label} (best)")
        ax.plot(hist['avg'], color=colors[key], linewidth=1, alpha=0.4, linestyle='--')
    ax.axhline(y=68.79, color='gold', linestyle='--', linewidth=2, label='Optimal (68.79)')
    ax.axhline(y=41.35, color='gray', linestyle=':', linewidth=1, label='CPG 12D (41.35)')
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax.set_title("Fitness Evolution: Energy-Constrained NN", fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 2: Force patterns for best condition
    ax = axes[0][1]
    # Find best overall
    best_key = max(results_dict.keys(), key=lambda k: results_dict[k]['displacement'])
    best_genes = results_dict[best_key]['genes']
    body = SoftBody()
    positions = body.rest_pos
    
    times_viz = [0.0, 0.5, 1.0, 1.5]
    colors_t = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    for ti, (t_val, col) in enumerate(zip(times_viz, colors_t)):
        forces = nn_generate_forces(best_genes, body, t_val)
        scale = 0.02
        for i in range(body.n):
            ax.arrow(positions[i, 0] + ti*0.03, positions[i, 1] + ti*0.03,
                    forces[i, 0] * scale, forces[i, 1] * scale,
                    head_width=0.04, head_length=0.015, fc=col, ec=col, alpha=0.5)
    ax.scatter(positions[:, 0], positions[:, 1], c='black', s=40, zorder=5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"Best ({results_dict[best_key]['label']}): Force Patterns", fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    
    # Panel 3: Method comparison bar chart
    ax = axes[1][0]
    methods = {}
    for key, data in results_dict.items():
        methods[f"NN v2\n{data['short_label']}"] = data['displacement']
    methods["NN v1\n(no constraint)"] = 261.65
    methods["Optimal\n(handcoded)"] = 68.79
    methods["CPG 12D"] = 41.35
    methods["CPG 5D"] = 35.35
    
    names = list(methods.keys())
    vals = list(methods.values())
    n_conds = len(results_dict)
    bar_colors = list(colors.values())[:n_conds] + ['#95a5a6', '#f1c40f', '#2ecc71', '#3498db']
    bars = ax.bar(range(len(names)), vals, color=bar_colors[:len(names)], alpha=0.85,
                 edgecolor='white', linewidth=2)
    for b, v in zip(bars, vals):
        y = max(v, 0) + 2
        ax.text(b.get_x()+b.get_width()/2, y, f'{v:.1f}',
               ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Forward Displacement")
    ax.set_title("All Methods Comparison", fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 4: Force over time for best condition (Fx for different particles)
    ax = axes[1][1]
    best_genes_plot = results_dict[best_key]['genes']
    body_viz = SoftBody()
    times_full = np.arange(N_STEPS) * DT
    particles_show = [0, 5, 17, 30, 35]
    colors_p = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#3498db']
    labels_p = ['P0 (corner)', 'P5 (edge)', 'P17 (center)', 'P30 (edge)', 'P35 (corner)']
    
    for pidx, col, lab in zip(particles_show, colors_p, labels_p):
        fx_hist = []
        for step in range(N_STEPS):
            forces = nn_generate_forces(best_genes_plot, body_viz, step * DT)
            fx_hist.append(forces[pidx, 0])
        ax.plot(times_full, fx_hist, color=col, linewidth=1.2, label=lab, alpha=0.8)
    
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Fx Force")
    ax.set_title(f"Best ({results_dict[best_key]['label']}): Force Output", fontweight='bold')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "evolution_nn_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

def main():
    print("=" * 60)
    print("  NN Controller v2: Energy-Constrained Locomotion")
    print("  Testing 3 constraint levels to force walking behavior")
    print(f"  Workers: {N_WORKERS} cores")
    print("=" * 60)
    
    t_total_start = time.time()
    
    # Condition A: Moderate constraint
    genes_A, fit_A, hist_A, time_A = evolve_condition(
        eval_condition_A, "Cond A: α=0.5, β=1.0 (moderate)", n_gens=200)
    
    # Condition B: Strong constraint  
    genes_B, fit_B, hist_B, time_B = evolve_condition(
        eval_condition_B, "Cond B: α=2.0, β=2.0 (strong)", n_gens=200)
    
    # Condition C: Very strong constraint
    genes_C, fit_C, hist_C, time_C = evolve_condition(
        eval_condition_C, "Cond C: α=5.0, β=3.0 (very strong)", n_gens=200)
    
    total_elapsed = time.time() - t_total_start
    
    # Get detailed metrics for each
    results = {}
    for key, genes, fit, hist, elapsed, label, short in [
        ('A', genes_A, fit_A, hist_A, time_A, 'α=0.5 β=1.0 (moderate)', 'α=0.5'),
        ('B', genes_B, fit_B, hist_B, time_B, 'α=2.0 β=2.0 (strong)', 'α=2.0'),
        ('C', genes_C, fit_C, hist_C, time_C, 'α=5.0 β=3.0 (very strong)', 'α=5.0'),
    ]:
        _, disp, energy, gait = evaluate_with_constraints(genes, alpha_energy=0, beta_gait=0)
        results[key] = {
            'label': label,
            'short_label': short,
            'genes': genes,
            'fitness': fit,
            'displacement': disp,
            'energy_norm': energy,
            'gait_reward': gait,
            'history': hist,
            'elapsed_min': elapsed / 60,
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Condition':<35} {'Fitness':>10} {'Disp':>10} {'Energy':>10}")
    print(f"  {'-'*65}")
    for key in ['A', 'B', 'C']:
        r = results[key]
        marker = " 🏆" if r['displacement'] > 68.79 else ""
        print(f"  {r['label']:<35} {r['fitness']:+10.2f} {r['displacement']:10.2f} {r['energy_norm']:10.4f}{marker}")
    print(f"  {'NN v1 (unconstrained)':<35} {'+261.65':>10} {'261.65':>10} {'~1.0':>10}")
    print(f"  {'Optimal (handcoded)':<35} {'68.79':>10} {'68.79':>10} {'???':>10}")
    print(f"  {'CPG 12D':<35} {'41.35':>10} {'41.35':>10} {'???':>10}")
    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print(f"{'='*60}")
    
    # Visualize
    fig_path = visualize_all(results)
    
    # Save results
    log_data = {
        "experiment": "NN Controller v2: Energy-Constrained Locomotion",
        "purpose": "Force genuine walking behavior instead of constant-push sliding",
        "nn_architecture": {"input": INPUT_SIZE, "hidden": HIDDEN_SIZE, "output": OUTPUT_SIZE,
                           "total_params": N_GENES},
        "conditions": {},
        "elapsed_min": round(total_elapsed / 60, 1),
        "figure": fig_path,
    }
    for key in ['A', 'B', 'C']:
        r = results[key]
        log_data["conditions"][key] = {
            "label": r['label'],
            "fitness": float(r['fitness']),
            "displacement": float(r['displacement']),
            "energy_norm": float(r['energy_norm']),
            "freq": float(abs(r['genes'][-1])),
            "elapsed_min": round(r['elapsed_min'], 1),
            "beaten_optimal": r['displacement'] > 68.79,
        }
    
    log_path = os.path.join(RESULTS_DIR, "evolution_nn_v2_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"  Log: {log_path}")
    
    # === BEEP when done! ===
    try:
        import winsound
        for _ in range(5):
            winsound.Beep(800, 300)
            time.sleep(0.2)
        print("\n  🔔 BEEP! Experiment complete!")
    except Exception:
        print("\n  ⚠️ Could not play beep sound")


if __name__ == "__main__":
    main()
