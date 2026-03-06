"""
CPG (Central Pattern Generator) Locomotion Evolution
=====================================================

Instead of 144 independent parameters, compresses the genome to
just 5 parameters that define a TRAVELING WAVE across the body:

  Phase[i] = 2π * freq * t + k_x * X_i + k_y * Y_i

Genes: [freq, amp_x, amp_y, k_x, k_y]

This mirrors biological CPGs (spinal cord walking circuits) where
a single wave propagates along the body with spatial gradients.

Hypothesis: By encoding spatial correlation structurally, the GA
will quickly find optimal locomotion that beats both the 144-dim
evolved solution AND the human-designed "optimal" controller.

CPU-only, 22-core parallel. Should converge in minutes.

Usage:
    python src/evolve_cpg.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os, time, json
from multiprocessing import Pool, cpu_count

N_WORKERS = max(1, cpu_count() - 2)

# === Physics Config ===
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
POP_SIZE = 200          # Larger pop since genome is tiny
N_GENERATIONS = 150     # More gens for thorough exploration
MUTATION_RATE = 0.3     # Higher mutation rate for 5D space
MUTATION_SIGMA = 0.2
ELITE_FRAC = 0.1
TOURNAMENT_SIZE = 3

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================================================
#  SOFT BODY
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
#  CPG GENOME: Only 5 parameters!
# ===================================================

class CPGGenome:
    """Central Pattern Generator: 5 genes define a traveling wave.
    
    genes[0] = freq     : oscillation frequency (Hz)
    genes[1] = amp_x    : force amplitude in X direction
    genes[2] = amp_y    : force amplitude in Y direction  
    genes[3] = k_x      : phase gradient along X (wave direction)
    genes[4] = k_y      : phase gradient along Y (wave direction)
    """
    N_GENES = 5
    
    def __init__(self):
        self.genes = np.array([
            np.random.uniform(0.5, 3.0),   # freq
            np.random.uniform(-1.0, 1.0),  # amp_x (will be scaled by BASE_AMP)
            np.random.uniform(-1.0, 1.0),  # amp_y
            np.random.uniform(-5.0, 5.0),  # k_x (phase gradient)
            np.random.uniform(-5.0, 5.0),  # k_y
        ])
        self.fitness = -np.inf

    def copy(self):
        g = CPGGenome.__new__(CPGGenome)
        g.genes = self.genes.copy()
        g.fitness = self.fitness
        return g

    def mutate(self, rate=MUTATION_RATE, sigma=MUTATION_SIGMA):
        for i in range(self.N_GENES):
            if np.random.random() < rate:
                self.genes[i] += np.random.randn() * sigma * [0.3, 0.5, 0.5, 1.0, 1.0][i]

    def generate_forces(self, body, t):
        """Generate CPG traveling wave forces for all particles."""
        BASE_AMP = 40.0
        n = body.n
        forces = np.zeros((n, 2))
        
        freq = abs(self.genes[0])
        amp_x = BASE_AMP * np.tanh(self.genes[1])
        amp_y = BASE_AMP * np.tanh(self.genes[2])
        k_x = self.genes[3]
        k_y = self.genes[4]
        
        for i in range(n):
            x_i = body.rest_pos[i, 0]
            y_i = body.rest_pos[i, 1]
            
            # CPG: Phase = 2π*freq*t + k_x*X + k_y*Y
            phase = 2 * np.pi * freq * t + k_x * x_i + k_y * y_i
            
            # Ground contact modulation
            ground_contact = 1.5 if body.pos[i, 1] < GROUND_Y + 0.3 else 0.5
            
            forces[i, 0] = amp_x * np.sin(phase) * ground_contact
            forces[i, 1] = amp_y * max(0, np.cos(phase)) * ground_contact
        
        return forces


def crossover(p1, p2):
    child = p1.copy()
    mask = np.random.random(child.genes.shape) < 0.5
    child.genes[mask] = p2.genes[mask]
    child.fitness = -np.inf
    return child


# ===================================================
#  FITNESS
# ===================================================

def evaluate_fitness_cpg(genes_array):
    """Evaluate CPG genome fitness (top-level for multiprocessing)."""
    g = CPGGenome.__new__(CPGGenome)
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
    print("="*60)
    print("  CPG Locomotion Evolution")
    print(f"  Population: {POP_SIZE}, Generations: {N_GENERATIONS}")
    print(f"  Genome: {CPGGenome.N_GENES} parameters (vs 144 in original!)")
    print(f"  [freq, amp_x, amp_y, k_x, k_y]")
    print(f"  Parallel workers: {N_WORKERS} cores")
    print("="*60)
    
    t0 = time.time()
    population = [CPGGenome() for _ in range(POP_SIZE)]
    
    history = {"best_fitness": [], "avg_fitness": [], "worst_fitness": [],
               "best_genes": []}
    
    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))
    
    for gen in range(N_GENERATIONS):
        gen_t0 = time.time()
        
        unevaluated = [(i, g) for i, g in enumerate(population) if g.fitness == -np.inf]
        if unevaluated:
            indices, genomes = zip(*unevaluated)
            genes_list = [g.genes.copy() for g in genomes]
            with Pool(N_WORKERS) as pool:
                fitnesses = pool.map(evaluate_fitness_cpg, genes_list)
            for idx, fit in zip(indices, fitnesses):
                population[idx].fitness = fit
        
        population.sort(key=lambda g: g.fitness, reverse=True)
        
        best = population[0]
        avg_fit = np.mean([g.fitness for g in population])
        worst_fit = population[-1].fitness
        
        history["best_fitness"].append(best.fitness)
        history["avg_fitness"].append(avg_fit)
        history["worst_fitness"].append(worst_fit)
        history["best_genes"].append(best.genes.tolist())
        
        gen_dt = time.time() - gen_t0
        if gen % 10 == 0 or gen == N_GENERATIONS - 1:
            print(f"  Gen {gen:3d}/{N_GENERATIONS}: best={best.fitness:+8.2f}  "
                  f"avg={avg_fit:+8.2f}  genes=[{', '.join(f'{g:.2f}' for g in best.genes)}]  "
                  f"[{gen_dt:.1f}s]")
        
        new_pop = [population[i].copy() for i in range(n_elite)]
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            child = crossover(p1, p2)
            child.mutate()
            new_pop.append(child)
        population = new_pop
    
    # Final eval
    for g in population:
        if g.fitness == -np.inf:
            g.fitness = evaluate_fitness_cpg(g.genes)
    population.sort(key=lambda g: g.fitness, reverse=True)
    
    total_time = time.time() - t0
    print(f"\n  Total time: {total_time/60:.1f} min")
    print(f"  Best fitness: {population[0].fitness:+.2f}")
    print(f"  Best genes: {population[0].genes}")
    
    return population[0], history, total_time


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(best, history, total_time):
    body = SoftBody()
    genes = best.genes
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"CPG Evolution: 5 Genes → Traveling Wave Locomotion\n"
                 f"Best fitness={best.fitness:.2f}  "
                 f"[freq={genes[0]:.2f}, amp_x={genes[1]:.2f}, amp_y={genes[2]:.2f}, "
                 f"k_x={genes[3]:.2f}, k_y={genes[4]:.2f}]",
                 fontsize=12, fontweight="bold")
    
    # Panel 1: Fitness evolution
    ax = axes[0][0]
    gens = range(len(history["best_fitness"]))
    ax.fill_between(gens, history["worst_fitness"], history["best_fitness"],
                   alpha=0.2, color='#3498db')
    ax.plot(gens, history["best_fitness"], 'b-', linewidth=2, label='Best')
    ax.plot(gens, history["avg_fitness"], 'g--', linewidth=1.5, label='Average')
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax.set_title("Fitness Evolution (5D CPG)", fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 2: Phase map on body (from CPG wave equation)
    ax = axes[0][1]
    positions = body.rest_pos
    k_x, k_y = genes[3], genes[4]
    phases = (k_x * positions[:, 0] + k_y * positions[:, 1]) % (2*np.pi)
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=phases,
                        cmap='hsv', s=200, edgecolors='black', linewidth=1,
                        vmin=0, vmax=2*np.pi)
    plt.colorbar(scatter, ax=ax, label='Phase (rad)')
    
    # Draw wave direction arrow
    wave_mag = np.sqrt(k_x**2 + k_y**2)
    if wave_mag > 0.1:
        cx, cy = np.mean(positions[:, 0]), np.mean(positions[:, 1])
        ax.annotate('', xy=(cx + k_x/wave_mag*0.8, cy + k_y/wave_mag*0.8),
                   xytext=(cx, cy),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        ax.text(cx + k_x/wave_mag*1.0, cy + k_y/wave_mag*1.0, 'Wave\nDirection',
               ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"CPG Phase Map (k_x={k_x:.2f}, k_y={k_y:.2f})", fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    
    # Panel 3: Comparison with all methods
    ax = axes[1][0]
    comparisons = {}
    
    # CPG evolved
    body_e = SoftBody(cx=0, cy=2.0)
    start = body_e.center_of_mass()[0]
    for step in range(N_STEPS):
        forces = best.generate_forces(body_e, step * DT)
        body_e.step(forces)
    comparisons["CPG\n(5 genes)"] = body_e.center_of_mass()[0] - start
    
    # Previous GA (144 genes) - approximate
    comparisons["GA 144D\n(prev)"] = 15.28
    
    # Independent baseline
    body_i = SoftBody(cx=0, cy=2.0)
    rng = np.random.RandomState(99)
    indep_phases = rng.uniform(0, 2*np.pi, N_PARTICLES)
    indep_freqs = rng.uniform(0.5, 4.0, N_PARTICLES)
    start = body_i.center_of_mass()[0]
    for step in range(N_STEPS):
        t = step * DT
        forces = np.zeros((N_PARTICLES, 2))
        for i in range(N_PARTICLES):
            forces[i, 0] = 40.0 * np.sin(2*np.pi * indep_freqs[i] * t + indep_phases[i])
            forces[i, 1] = 20.0 * np.sin(2*np.pi * indep_freqs[i] * 1.3 * t + indep_phases[i] + 1.0)
        body_i.step(forces)
    comparisons["Independent\n(seizure)"] = body_i.center_of_mass()[0] - start
    
    names = list(comparisons.keys())
    vals = list(comparisons.values())
    colors = ['#e74c3c', '#f39c12', '#95a5a6']
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85,
                 edgecolor='white', linewidth=2)
    for b, v in zip(bars, vals):
        y = max(v, 0) + 0.5
        ax.text(b.get_x()+b.get_width()/2, y, f'{v:.1f}',
               ha='center', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Forward Displacement")
    ax.set_title("CPG vs Previous Methods", fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 4: Gene evolution over generations
    ax = axes[1][1]
    gene_history = np.array(history["best_genes"])
    labels_g = ['freq', 'amp_x', 'amp_y', 'k_x', 'k_y']
    colors_g = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    for i, (lab, col) in enumerate(zip(labels_g, colors_g)):
        ax.plot(gene_history[:, i], color=col, linewidth=1.5, label=lab, alpha=0.8)
    ax.set_xlabel("Generation"); ax.set_ylabel("Gene Value")
    ax.set_title("Gene Evolution Over Generations", fontweight='bold')
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "evolution_cpg.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


def main():
    best, history, total_time = evolve()
    
    genes = best.genes
    print(f"\n{'='*60}")
    print(f"  CPG ANALYSIS")
    print(f"{'='*60}")
    print(f"  Evolved traveling wave parameters:")
    print(f"    freq  = {genes[0]:.3f} Hz")
    print(f"    amp_x = {genes[1]:.3f} (tanh → {np.tanh(genes[1])*40:.1f} N)")
    print(f"    amp_y = {genes[2]:.3f} (tanh → {np.tanh(genes[2])*40:.1f} N)")
    print(f"    k_x   = {genes[3]:.3f} rad/unit (X phase gradient)")
    print(f"    k_y   = {genes[4]:.3f} rad/unit (Y phase gradient)")
    
    wave_dir = np.degrees(np.arctan2(genes[4], genes[3]))
    wave_speed = abs(genes[0]) / max(0.01, np.sqrt(genes[3]**2 + genes[4]**2))
    print(f"    Wave direction: {wave_dir:.1f}°")
    print(f"    Wave speed: {wave_speed:.2f} units/s")
    print(f"    Spatial correlation: STRUCTURAL (r≈1.0 by design!)")
    print(f"{'='*60}")
    
    fig_path = visualize(best, history, total_time)
    
    results = {
        "experiment": "CPG Locomotion Evolution",
        "genome_size": CPGGenome.N_GENES,
        "population_size": POP_SIZE,
        "n_generations": N_GENERATIONS,
        "best_fitness": float(best.fitness),
        "best_genes": {
            "freq": float(genes[0]),
            "amp_x": float(genes[1]),
            "amp_y": float(genes[2]),
            "k_x": float(genes[3]),
            "k_y": float(genes[4]),
        },
        "elapsed_min": round(total_time / 60, 1),
        "fitness_history": {
            "best": history["best_fitness"],
            "avg": history["avg_fitness"],
        },
        "figure": fig_path,
    }
    log_path = os.path.join(RESULTS_DIR, "evolution_cpg_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
