"""
Neural Network Controller Evolution
=====================================

Replace CPG (fixed sin/cos wave) with a small neural network that
takes time as input and outputs force vectors for all particles.

Architecture:
  Input (4):  [sin(2π·f·t), cos(2π·f·t), norm_x_i, norm_y_i]
  Hidden (16): tanh activation
  Output (2):  [Fx, Fy] per particle

Total parameters: NN weights + biases + freq = (4*16 + 16) + (16*2 + 2) + 1 = 99 + 34 + 1 = 131
                  Still << 144 (raw GA), but >> 5 (CPG)

Key advantage over CPG:
  - Can express DISCRETE switching (tanh saturates → step-like behavior)
  - Can learn nonlinear ground-contact strategies
  - Time input via sin/cos avoids extrapolation issues

Target: Beat handcoded Optimal (68.79)!

CPU-only, 22-core parallel.

Usage:
    python src/evolve_nn.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os, time, json
from multiprocessing import Pool, cpu_count

N_WORKERS = max(1, cpu_count() - 2)

# === Physics Config (same as evolve_cpg.py) ===
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

# === GA Config ===
POP_SIZE = 300          # Larger pop for bigger genome
N_GENERATIONS = 300     # More generations for NN convergence
MUTATION_RATE = 0.15    # Lower per-gene rate (more genes)
MUTATION_SIGMA = 0.3    # Moderate sigma
ELITE_FRAC = 0.05       # Keep top 5%
TOURNAMENT_SIZE = 5     # More selective pressure

# === NN Config ===
HIDDEN_SIZE = 16
INPUT_SIZE = 4     # [sin(2π·f·t), cos(2π·f·t), norm_x_i, norm_y_i]
OUTPUT_SIZE = 2    # [Fx, Fy]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================================================
#  SOFT BODY (identical to evolve_cpg.py)
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
#  NN GENOME
# ===================================================

class NNGenome:
    """Small neural network controller.
    
    Architecture: Input(4) -> Hidden(16, tanh) -> Output(2)
    
    Input per particle:
        [sin(2π·freq·t), cos(2π·freq·t), norm_x_i, norm_y_i]
        where norm_x_i, norm_y_i are normalized rest positions [-1, 1]
    
    Output per particle:
        [Fx, Fy] force vector (scaled by BASE_AMP)
    
    Genome = flattened [W1, b1, W2, b2, freq]
    """
    
    # Compute total genome size
    N_W1 = INPUT_SIZE * HIDDEN_SIZE   # 4*16 = 64
    N_B1 = HIDDEN_SIZE                # 16
    N_W2 = HIDDEN_SIZE * OUTPUT_SIZE  # 16*2 = 32
    N_B2 = OUTPUT_SIZE                # 2
    N_FREQ = 1                        # oscillation frequency
    N_GENES = N_W1 + N_B1 + N_W2 + N_B2 + N_FREQ  # 64+16+32+2+1 = 115
    
    def __init__(self):
        # Xavier initialization for weights, zeros for biases
        scale_w1 = np.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE))
        scale_w2 = np.sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE))
        
        w1 = np.random.randn(self.N_W1) * scale_w1
        b1 = np.zeros(self.N_B1)
        w2 = np.random.randn(self.N_W2) * scale_w2
        b2 = np.zeros(self.N_B2)
        freq = np.array([np.random.uniform(0.5, 3.0)])
        
        self.genes = np.concatenate([w1, b1, w2, b2, freq])
        self.fitness = -np.inf
    
    def copy(self):
        g = NNGenome.__new__(NNGenome)
        g.genes = self.genes.copy()
        g.fitness = self.fitness
        return g
    
    def mutate(self, rate=MUTATION_RATE, sigma=MUTATION_SIGMA):
        mask = np.random.random(self.N_GENES) < rate
        mutations = np.random.randn(self.N_GENES) * sigma
        # Scale mutations differently for weights vs freq
        # Last gene is freq - mutate more gently
        mutations[-1] *= 0.3
        self.genes[mask] += mutations[mask]
    
    def _unpack(self):
        """Unpack genome into weight matrices."""
        idx = 0
        W1 = self.genes[idx:idx+self.N_W1].reshape(INPUT_SIZE, HIDDEN_SIZE)
        idx += self.N_W1
        b1 = self.genes[idx:idx+self.N_B1]
        idx += self.N_B1
        W2 = self.genes[idx:idx+self.N_W2].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
        idx += self.N_W2
        b2 = self.genes[idx:idx+self.N_B2]
        idx += self.N_B2
        freq = abs(self.genes[idx])
        return W1, b1, W2, b2, freq
    
    def generate_forces(self, body, t):
        """Generate NN-controlled forces for all particles."""
        BASE_AMP = 40.0
        n = body.n
        forces = np.zeros((n, 2))
        
        W1, b1, W2, b2, freq = self._unpack()
        
        # Time signal (shared across all particles)
        sin_t = np.sin(2 * np.pi * freq * t)
        cos_t = np.cos(2 * np.pi * freq * t)
        
        # Precompute normalized positions [-1, 1]
        x_min = body.rest_pos[:, 0].min()
        x_max = body.rest_pos[:, 0].max()
        y_min = body.rest_pos[:, 1].min()
        y_max = body.rest_pos[:, 1].max()
        x_range = x_max - x_min + 1e-8
        y_range = y_max - y_min + 1e-8
        
        for i in range(n):
            # Normalize position to [-1, 1]
            nx = 2 * (body.rest_pos[i, 0] - x_min) / x_range - 1
            ny = 2 * (body.rest_pos[i, 1] - y_min) / y_range - 1
            
            # Input vector
            inp = np.array([sin_t, cos_t, nx, ny])
            
            # Forward pass: Input -> Hidden (tanh) -> Output (tanh)
            hidden = np.tanh(inp @ W1 + b1)
            output = np.tanh(hidden @ W2 + b2)
            
            # Ground contact modulation (same as CPG)
            ground_contact = 1.5 if body.pos[i, 1] < GROUND_Y + 0.3 else 0.5
            
            # Scale output to force
            forces[i, 0] = BASE_AMP * output[0] * ground_contact
            forces[i, 1] = BASE_AMP * max(0, output[1]) * ground_contact  # Y: positive only (push up)
        
        return forces


def crossover(p1, p2):
    """Uniform crossover with segment awareness."""
    child = p1.copy()
    # Use segment-aware crossover: swap entire weight matrices sometimes
    if np.random.random() < 0.3:
        # Swap entire segments
        segment_boundaries = [0, NNGenome.N_W1, NNGenome.N_W1 + NNGenome.N_B1,
                             NNGenome.N_W1 + NNGenome.N_B1 + NNGenome.N_W2,
                             NNGenome.N_W1 + NNGenome.N_B1 + NNGenome.N_W2 + NNGenome.N_B2,
                             NNGenome.N_GENES]
        for s in range(len(segment_boundaries) - 1):
            if np.random.random() < 0.5:
                child.genes[segment_boundaries[s]:segment_boundaries[s+1]] = \
                    p2.genes[segment_boundaries[s]:segment_boundaries[s+1]]
    else:
        # Uniform crossover
        mask = np.random.random(child.genes.shape) < 0.5
        child.genes[mask] = p2.genes[mask]
    child.fitness = -np.inf
    return child


# ===================================================
#  FITNESS
# ===================================================

def evaluate_fitness_nn(genes_array):
    """Evaluate NN genome fitness (top-level for multiprocessing)."""
    g = NNGenome.__new__(NNGenome)
    g.genes = genes_array.copy()
    g.fitness = -np.inf
    
    body = SoftBody(cx=0, cy=2.0)
    start_com = body.center_of_mass()
    
    for step in range(N_STEPS):
        t = step * DT
        forces = g.generate_forces(body, t)
        body.step(forces)
    
    end_com = body.center_of_mass()
    displacement = end_com[0] - start_com[0]
    
    extent = np.max(body.pos, axis=0) - np.min(body.pos, axis=0)
    spread_penalty = max(0, extent[0] - 5.0) * 2.0 + max(0, extent[1] - 5.0) * 2.0
    below_ground = np.sum(body.pos[:, 1] < GROUND_Y - 1.0)
    ground_penalty = below_ground * 0.5
    
    return displacement - spread_penalty - ground_penalty


# ===================================================
#  GA
# ===================================================

def tournament_select(population, k=TOURNAMENT_SIZE):
    candidates = np.random.choice(len(population), k, replace=False)
    best = max(candidates, key=lambda i: population[i].fitness)
    return population[best]


def evolve():
    print("=" * 60)
    print("  Neural Network Controller Evolution")
    print(f"  Population: {POP_SIZE}, Generations: {N_GENERATIONS}")
    print(f"  NN Architecture: Input({INPUT_SIZE}) -> Hidden({HIDDEN_SIZE}, tanh) -> Output({OUTPUT_SIZE})")
    print(f"  Genome: {NNGenome.N_GENES} parameters")
    print(f"  Target: Handcoded Optimal = 68.79")
    print(f"  Parallel workers: {N_WORKERS} cores")
    print("=" * 60)
    
    t0 = time.time()
    population = [NNGenome() for _ in range(POP_SIZE)]
    
    history = {"best_fitness": [], "avg_fitness": [], "worst_fitness": [],
               "best_genes_hash": []}
    
    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))
    stagnation_count = 0
    prev_best = -np.inf
    
    for gen in range(N_GENERATIONS):
        gen_t0 = time.time()
        
        # Evaluate unevaluated individuals
        unevaluated = [(i, g) for i, g in enumerate(population) if g.fitness == -np.inf]
        if unevaluated:
            indices, genomes = zip(*unevaluated)
            genes_list = [g.genes.copy() for g in genomes]
            with Pool(N_WORKERS) as pool:
                fitnesses = pool.map(evaluate_fitness_nn, genes_list)
            for idx, fit in zip(indices, fitnesses):
                population[idx].fitness = fit
        
        population.sort(key=lambda g: g.fitness, reverse=True)
        
        best = population[0]
        avg_fit = np.mean([g.fitness for g in population])
        worst_fit = population[-1].fitness
        
        history["best_fitness"].append(best.fitness)
        history["avg_fitness"].append(avg_fit)
        history["worst_fitness"].append(worst_fit)
        history["best_genes_hash"].append(hash(best.genes.tobytes()) % 10000)
        
        # Stagnation detection
        if best.fitness > prev_best + 0.1:
            stagnation_count = 0
            prev_best = best.fitness
        else:
            stagnation_count += 1
        
        gen_dt = time.time() - gen_t0
        if gen % 10 == 0 or gen == N_GENERATIONS - 1:
            marker = " 🏆" if best.fitness > 68.79 else ""
            freq = abs(best.genes[-1])
            print(f"  Gen {gen:3d}/{N_GENERATIONS}: best={best.fitness:+8.2f}  "
                  f"avg={avg_fit:+8.2f}  freq={freq:.2f}Hz  "
                  f"stag={stagnation_count}  [{gen_dt:.1f}s]{marker}")
        
        # Adaptive mutation based on stagnation
        current_sigma = MUTATION_SIGMA
        current_rate = MUTATION_RATE
        if stagnation_count > 20:
            current_sigma = MUTATION_SIGMA * 1.5  # Explore more
            current_rate = MUTATION_RATE * 1.5
        if stagnation_count > 50:
            current_sigma = MUTATION_SIGMA * 2.0  # Major exploration
            current_rate = MUTATION_RATE * 2.0
        
        # Early stopping if we beat Optimal
        if best.fitness > 68.79 and gen > 50:
            print(f"\n  🏆 OPTIMAL BEATEN at gen {gen}! {best.fitness:.2f} > 68.79")
            break
        
        # Build next generation
        new_pop = [population[i].copy() for i in range(n_elite)]
        
        # Inject fresh random individuals if stagnating
        n_fresh = 0
        if stagnation_count > 30:
            n_fresh = int(POP_SIZE * 0.1)
            for _ in range(n_fresh):
                new_pop.append(NNGenome())
        
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            child = crossover(p1, p2)
            child.mutate(rate=current_rate, sigma=current_sigma)
            new_pop.append(child)
        population = new_pop
    
    # Final eval
    for g in population:
        if g.fitness == -np.inf:
            g.fitness = evaluate_fitness_nn(g.genes)
    population.sort(key=lambda g: g.fitness, reverse=True)
    
    total_time = time.time() - t0
    print(f"\n  Total time: {total_time/60:.1f} min")
    print(f"  Best fitness: {population[0].fitness:+.2f}")
    
    return population[0], history, total_time


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(best, history, total_time):
    body = SoftBody()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"NN Controller Evolution: {NNGenome.N_GENES} params → Learned Locomotion\n"
                 f"Best fitness={best.fitness:.2f}  |  Target: Optimal=68.79  |  "
                 f"Time: {total_time/60:.1f} min",
                 fontsize=12, fontweight="bold")
    
    # Panel 1: Fitness evolution
    ax = axes[0][0]
    gens = range(len(history["best_fitness"]))
    ax.fill_between(gens, history["worst_fitness"], history["best_fitness"],
                   alpha=0.15, color='#e74c3c')
    ax.plot(gens, history["best_fitness"], '#e74c3c', linewidth=2, label='Best (NN)')
    ax.plot(gens, history["avg_fitness"], '#f39c12', linewidth=1.5, label='Average', alpha=0.7)
    ax.axhline(y=68.79, color='gold', linestyle='--', linewidth=2, label='Optimal (68.79)')
    ax.axhline(y=35.35, color='gray', linestyle=':', linewidth=1.5, label='CPG 5D (35.35)')
    ax.axhline(y=41.35, color='green', linestyle=':', linewidth=1.5, label='CPG 12D (41.35)')
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax.set_title("NN Controller Fitness Evolution", fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 2: Force pattern visualization at t=0, 1, 2, 3 seconds
    ax = axes[0][1]
    positions = body.rest_pos
    times = [0.0, 1.0, 2.0, 3.0]
    colors_t = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    
    for ti, (t_val, col) in enumerate(zip(times, colors_t)):
        forces = best.generate_forces(body, t_val)
        scale = 0.02
        for i in range(body.n):
            ax.arrow(positions[i, 0] + ti*0.05, positions[i, 1] + ti*0.05,
                    forces[i, 0] * scale, forces[i, 1] * scale,
                    head_width=0.05, head_length=0.02, fc=col, ec=col, alpha=0.5)
    
    ax.scatter(positions[:, 0], positions[:, 1], c='black', s=50, zorder=5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Force Patterns at t=0,1,2,3s", fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    
    # Panel 3: Method comparison
    ax = axes[1][0]
    comparisons = {}
    
    # NN evolved
    body_e = SoftBody(cx=0, cy=2.0)
    start = body_e.center_of_mass()[0]
    for step in range(N_STEPS):
        forces = best.generate_forces(body_e, step * DT)
        body_e.step(forces)
    comparisons["NN\n(115 genes)"] = body_e.center_of_mass()[0] - start
    
    # References
    comparisons["Optimal\n(handcoded)"] = 68.79
    comparisons["CPG 12D\n(prev best)"] = 41.35
    comparisons["CPG 5D"] = 35.35
    comparisons["GA 144D"] = 15.28
    
    names = list(comparisons.keys())
    vals = list(comparisons.values())
    colors_bar = ['#e74c3c', '#f1c40f', '#2ecc71', '#3498db', '#95a5a6']
    bars = ax.bar(range(len(names)), vals, color=colors_bar, alpha=0.85,
                 edgecolor='white', linewidth=2)
    for b, v in zip(bars, vals):
        y = max(v, 0) + 1.0
        ax.text(b.get_x()+b.get_width()/2, y, f'{v:.1f}',
               ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Forward Displacement")
    ax.set_title("NN Controller vs All Methods", fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 4: Force magnitude over time for select particles
    ax = axes[1][1]
    times_full = np.arange(N_STEPS) * DT
    particles_show = [0, 5, 17, 30, 35]  # Corners + center
    colors_p = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#3498db']
    particle_labels = ['P0 (corner)', 'P5 (edge)', 'P17 (center)', 'P30 (edge)', 'P35 (corner)']
    
    body_viz = SoftBody()
    for pi, (pidx, col, lab) in enumerate(zip(particles_show, colors_p, particle_labels)):
        fx_history = []
        for step in range(N_STEPS):
            t = step * DT
            forces = best.generate_forces(body_viz, t)
            fx_history.append(forces[pidx, 0])
        ax.plot(times_full, fx_history, color=col, linewidth=1.2, label=lab, alpha=0.8)
    
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Fx Force")
    ax.set_title("NN Force Output Over Time", fontweight='bold')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "evolution_nn.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


def main():
    best, history, total_time = evolve()
    
    _, _, _, _, freq = best._unpack()
    print(f"\n{'='*60}")
    print(f"  NN CONTROLLER ANALYSIS")
    print(f"{'='*60}")
    print(f"  Architecture: Input(4) -> Hidden({HIDDEN_SIZE}) -> Output(2)")
    print(f"  Total parameters: {NNGenome.N_GENES}")
    print(f"  Evolved frequency: {freq:.3f} Hz")
    print(f"  Best fitness: {best.fitness:.2f}")
    print(f"  Target (Optimal): 68.79")
    print(f"  {'🏆 BEATEN!' if best.fitness > 68.79 else '❌ Not beaten yet'}")
    print(f"{'='*60}")
    
    fig_path = visualize(best, history, total_time)
    
    result = best.fitness > 68.79
    results = {
        "experiment": "NN Controller Evolution",
        "nn_architecture": {
            "input": INPUT_SIZE,
            "hidden": HIDDEN_SIZE,
            "output": OUTPUT_SIZE,
            "total_params": NNGenome.N_GENES,
        },
        "ga_config": {
            "population": POP_SIZE,
            "generations": N_GENERATIONS,
            "mutation_rate": MUTATION_RATE,
            "mutation_sigma": MUTATION_SIGMA,
        },
        "best_fitness": float(best.fitness),
        "optimal_target": 68.79,
        "beaten_optimal": result,
        "evolved_freq": float(freq),
        "elapsed_min": round(total_time / 60, 1),
        "fitness_history": {
            "best": [float(x) for x in history["best_fitness"]],
            "avg": [float(x) for x in history["avg_fitness"]],
        },
        "figure": fig_path,
    }
    log_path = os.path.join(RESULTS_DIR, "evolution_nn_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
