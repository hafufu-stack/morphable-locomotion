"""
Evolutionary Discovery with Energy Penalty
==========================================

Same as evolve_locomotion.py but with an ENERGY PENALTY in the
fitness function. The hypothesis: when wasting energy has a cost,
evolution will discover that spatially-correlated signals (waves)
are the most efficient locomotion strategy.

  fitness = displacement - alpha * total_energy

This mirrors biological evolution: animals evolved coordinated gaits
because synchronizing muscle contractions is energy-efficient.

CPU-only. With multiprocessing (24 cores), runs in ~11 minutes.

Usage:
    python src/evolve_energy_penalty.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os, time, json, copy
from multiprocessing import Pool, cpu_count

N_WORKERS = max(1, cpu_count() - 2)  # Leave 2 cores for GPU work + system

# === Physics Config (same as morphable_locomotion.py) ===
N_PARTICLES = 36
GRID_SIZE = 6
DT = 0.015
N_STEPS = 400       # Shorter than demo for faster evolution
SPRING_K = 20.0
SPRING_DAMP = 1.0
DRAG = 0.3
SIGNAL_FREQ = 1.5
GROUND_Y = -0.5
GROUND_K = 500.0
GRAVITY = -8.0

# === GA Config ===
POP_SIZE = 120          # Larger population (24 cores can handle it)
N_GENERATIONS = 100     # More generations for better convergence
MUTATION_RATE = 0.15    # Per-gene mutation probability
MUTATION_SIGMA = 0.3    # Mutation step size
ELITE_FRAC = 0.1        # Top fraction kept unchanged
TOURNAMENT_SIZE = 3     # Tournament selection size

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================================================
#  SOFT BODY (simplified for speed)
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
#  GENOME: Per-particle oscillator parameters
# ===================================================

class Genome:
    """Each particle has: [phase_offset, freq_scale, amp_x, amp_y]"""
    GENES_PER_PARTICLE = 4
    
    def __init__(self, n_particles=N_PARTICLES):
        self.n = n_particles
        # Random initialization
        self.genes = np.random.randn(n_particles, self.GENES_PER_PARTICLE) * 0.5
        # genes[:, 0] = phase_offset (will be multiplied by 2π)
        # genes[:, 1] = freq_scale (centered around 1.0)
        # genes[:, 2] = amp_x (force amplitude X)
        # genes[:, 3] = amp_y (force amplitude Y)
        self.fitness = -np.inf

    def copy(self):
        g = Genome(self.n)
        g.genes = self.genes.copy()
        g.fitness = self.fitness
        return g

    def mutate(self, rate=MUTATION_RATE, sigma=MUTATION_SIGMA):
        mask = np.random.random(self.genes.shape) < rate
        self.genes += mask * np.random.randn(*self.genes.shape) * sigma

    def generate_forces(self, body, t):
        """Generate forces from genome-encoded oscillators."""
        n = self.n
        forces = np.zeros((n, 2))
        
        base_amp = 40.0  # Same as manual simulation
        
        for i in range(n):
            phase_offset = self.genes[i, 0] * 2 * np.pi
            freq = SIGNAL_FREQ * (1.0 + 0.5 * np.tanh(self.genes[i, 1]))
            amp_x = base_amp * np.tanh(self.genes[i, 2])
            amp_y = base_amp * np.tanh(self.genes[i, 3])
            
            phase = 2 * np.pi * freq * t + phase_offset
            
            # Ground contact boost
            ground_contact = 1.5 if body.pos[i, 1] < GROUND_Y + 0.3 else 0.5
            
            forces[i, 0] = amp_x * np.sin(phase) * ground_contact
            forces[i, 1] = amp_y * max(0, np.cos(phase)) * ground_contact
        
        return forces


def crossover(parent1, parent2):
    """Uniform crossover between two genomes."""
    child = parent1.copy()
    mask = np.random.random(child.genes.shape) < 0.5
    child.genes[mask] = parent2.genes[mask]
    child.fitness = -np.inf
    return child


# === Energy Penalty Config ===
ENERGY_PENALTY_ALPHA = 0.003  # 10x stronger energy cost

def evaluate_fitness(genome, seed=None):
    """Fitness = displacement - alpha * total_energy.
    Energy = sum of squared forces over all timesteps.
    This penalizes 'seizure' gaits that waste energy."""
    if seed is not None:
        np.random.seed(seed)
    
    body = SoftBody(cx=0, cy=2.0)
    start_com = body.center_of_mass()
    
    total_energy = 0.0
    
    for step in range(N_STEPS):
        t = step * DT
        forces = genome.generate_forces(body, t)
        total_energy += np.sum(forces**2)  # Energy ∝ force²
        body.step(forces)
    
    end_com = body.center_of_mass()
    displacement = end_com[0] - start_com[0]
    
    # Penalty for body disintegration
    extent = np.max(body.pos, axis=0) - np.min(body.pos, axis=0)
    spread_penalty = max(0, extent[0] - 5.0) * 2.0 + max(0, extent[1] - 5.0) * 2.0
    
    # Penalty for falling below ground
    below_ground = np.sum(body.pos[:, 1] < GROUND_Y - 1.0)
    ground_penalty = below_ground * 0.5
    
    # Energy-efficient locomotion!
    energy_penalty = ENERGY_PENALTY_ALPHA * total_energy
    
    fitness = displacement - spread_penalty - ground_penalty - energy_penalty
    return fitness


def _eval_genes(genes_array):
    """Top-level function for multiprocessing (must be picklable).
    Takes a numpy array, creates a Genome, evaluates and returns fitness."""
    g = Genome.__new__(Genome)
    g.n = N_PARTICLES
    g.genes = genes_array
    g.fitness = -np.inf
    return evaluate_fitness(g, seed=42)


# ===================================================
#  GENETIC ALGORITHM
# ===================================================

def tournament_select(population, k=TOURNAMENT_SIZE):
    """Tournament selection."""
    candidates = np.random.choice(len(population), k, replace=False)
    best = max(candidates, key=lambda i: population[i].fitness)
    return population[best]


def evolve():
    """Main evolutionary loop with parallel fitness evaluation."""
    print("="*60)
    print("  Evolutionary Discovery of Locomotion Patterns")
    print(f"  Population: {POP_SIZE}, Generations: {N_GENERATIONS}")
    print(f"  Genome: {N_PARTICLES} particles x {Genome.GENES_PER_PARTICLE} genes = {N_PARTICLES*Genome.GENES_PER_PARTICLE} params")
    print("  [ENERGY PENALTY MODE] alpha =", ENERGY_PENALTY_ALPHA)
    print("="*60)
    
    t0 = time.time()
    
    # Initialize population
    population = [Genome() for _ in range(POP_SIZE)]
    
    history = {
        "best_fitness": [],
        "avg_fitness": [],
        "worst_fitness": [],
        "best_genome_phases": [],  # Track how phases evolve
    }
    
    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))
    
    for gen in range(N_GENERATIONS):
        gen_t0 = time.time()
        
        # Evaluate fitness (PARALLEL)
        unevaluated = [(i, g) for i, g in enumerate(population) if g.fitness == -np.inf]
        if unevaluated:
            indices, genomes = zip(*unevaluated)
            genes_list = [g.genes.copy() for g in genomes]
            with Pool(N_WORKERS) as pool:
                fitnesses = pool.map(_eval_genes, genes_list)
            for idx, fit in zip(indices, fitnesses):
                population[idx].fitness = fit
        
        # Sort by fitness (descending)
        population.sort(key=lambda g: g.fitness, reverse=True)
        
        best = population[0].fitness
        avg = np.mean([g.fitness for g in population])
        worst = population[-1].fitness
        
        history["best_fitness"].append(best)
        history["avg_fitness"].append(avg)
        history["worst_fitness"].append(worst)
        
        # Save best genome's phase pattern
        best_phases = population[0].genes[:, 0] * 2 * np.pi
        history["best_genome_phases"].append(best_phases.tolist())
        
        gen_dt = time.time() - gen_t0
        print(f"  Gen {gen:3d}/{N_GENERATIONS}: best={best:+7.2f}  avg={avg:+7.2f}  "
              f"worst={worst:+7.2f}  [{gen_dt:.1f}s]")
        
        # Create next generation
        new_pop = []
        
        # Elitism: keep top individuals
        for i in range(n_elite):
            new_pop.append(population[i].copy())
        
        # Fill rest with crossover + mutation
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            child = crossover(p1, p2)
            child.mutate()
            new_pop.append(child)
        
        population = new_pop
    
    # Final evaluation
    for genome in population:
        if genome.fitness == -np.inf:
            genome.fitness = evaluate_fitness(genome, seed=42)
    population.sort(key=lambda g: g.fitness, reverse=True)
    
    total_time = time.time() - t0
    print(f"\n  Total time: {total_time/60:.1f} min")
    print(f"  Best fitness: {population[0].fitness:+.2f}")
    
    return population[0], history, total_time


# ===================================================
#  ANALYSIS: Did evolution discover spatial correlation?
# ===================================================

def analyze_correlation(best_genome):
    """Analyze whether the evolved phase pattern shows spatial structure."""
    body = SoftBody()
    phases = best_genome.genes[:, 0] * 2 * np.pi  # Phase offsets
    positions = body.rest_pos
    
    n = len(phases)
    distances = []
    phase_diffs = []
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            # Circular phase difference (0 to π)
            d_phase = abs(phases[i] - phases[j]) % (2*np.pi)
            if d_phase > np.pi:
                d_phase = 2*np.pi - d_phase
            distances.append(dist)
            phase_diffs.append(d_phase)
    
    distances = np.array(distances)
    phase_diffs = np.array(phase_diffs)
    
    # Correlation between distance and phase difference
    corr = np.corrcoef(distances, phase_diffs)[0, 1]
    
    return distances, phase_diffs, corr, phases


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(best_genome, history, total_time):
    """Create comprehensive visualization of evolutionary results."""
    body = SoftBody()
    distances, phase_diffs, corr, phases = analyze_correlation(best_genome)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Energy-Penalized Evolution: Does Efficiency Create Correlation?\n"
                 f"Spatial correlation r={corr:.3f} (alpha={ENERGY_PENALTY_ALPHA})",
                 fontsize=13, fontweight="bold")
    
    # Panel 1: Fitness over generations
    ax = axes[0][0]
    gens = range(len(history["best_fitness"]))
    ax.fill_between(gens, history["worst_fitness"], history["best_fitness"],
                   alpha=0.2, color='#3498db')
    ax.plot(gens, history["best_fitness"], 'b-', linewidth=2, label='Best')
    ax.plot(gens, history["avg_fitness"], 'g--', linewidth=1.5, label='Average')
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Fitness (displacement)", fontsize=11)
    ax.set_title("Fitness Evolution", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 2: Distance vs Phase difference (key discovery!)
    ax = axes[0][1]
    ax.scatter(distances, phase_diffs, alpha=0.3, s=10, c='#e74c3c')
    # Bin and show trend
    n_bins = 8
    bins = np.linspace(distances.min(), distances.max(), n_bins+1)
    bin_centers = []
    bin_means = []
    for i in range(n_bins):
        mask = (distances >= bins[i]) & (distances < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(phase_diffs[mask].mean())
    ax.plot(bin_centers, bin_means, 'ko-', linewidth=2, markersize=8, label='Trend')
    ax.set_xlabel("Particle Distance", fontsize=11)
    ax.set_ylabel("Phase Difference (rad)", fontsize=11)
    ax.set_title(f"Spatial Correlation: r={corr:.3f}\n"
                 f"{'DISCOVERED: Nearby = similar phase!' if corr > 0.1 else 'No spatial pattern'}",
                 fontsize=11, fontweight='bold',
                 color='green' if corr > 0.1 else 'red')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 3: Phase map on body
    ax = axes[1][0]
    positions = body.rest_pos
    # Normalize phases to [0, 2π] for colormap
    norm_phases = phases % (2*np.pi)
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=norm_phases,
                        cmap='hsv', s=200, edgecolors='black', linewidth=1,
                        vmin=0, vmax=2*np.pi)
    plt.colorbar(scatter, ax=ax, label='Phase offset (rad)')
    ax.set_xlabel("X position", fontsize=11)
    ax.set_ylabel("Y position", fontsize=11)
    ax.set_title("Evolved Phase Pattern on Body\n(similar colors = synchronized)",
                fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Panel 4: Comparison with baselines
    ax = axes[1][1]
    # Run the evolved genome and baselines
    from morphable_locomotion import generate_correlated_forces, SoftBody as ManualBody
    
    comparisons = {}
    # Evolved
    body_e = SoftBody(cx=0, cy=2.0)
    start = body_e.center_of_mass()[0]
    for step in range(N_STEPS):
        forces = best_genome.generate_forces(body_e, step * DT)
        body_e.step(forces)
    comparisons["Evolved (GA)"] = body_e.center_of_mass()[0] - start
    
    # Independent baseline
    body_i = SoftBody(cx=0, cy=2.0)
    np.random.seed(99)
    indep_phases = np.random.uniform(0, 2*np.pi, N_PARTICLES)
    indep_freqs = np.random.uniform(0.5, 4.0, N_PARTICLES)
    start = body_i.center_of_mass()[0]
    for step in range(N_STEPS):
        t = step * DT
        forces = np.zeros((N_PARTICLES, 2))
        for i in range(N_PARTICLES):
            forces[i, 0] = 40.0 * np.sin(2*np.pi * indep_freqs[i] * t + indep_phases[i])
            forces[i, 1] = 20.0 * np.sin(2*np.pi * indep_freqs[i] * 1.3 * t + indep_phases[i] + 1.0)
        body_i.step(forces)
    comparisons["Independent\n(random)"] = body_i.center_of_mass()[0] - start
    
    # Uniform phase (all same)
    body_u = SoftBody(cx=0, cy=2.0)
    start = body_u.center_of_mass()[0]
    for step in range(N_STEPS):
        t = step * DT
        forces = np.zeros((N_PARTICLES, 2))
        phase = 2 * np.pi * SIGNAL_FREQ * t
        for i in range(N_PARTICLES):
            gc = 1.5 if body_u.pos[i, 1] < GROUND_Y + 0.3 else 0.5
            forces[i, 0] = 40.0 * np.sin(phase) * gc
            forces[i, 1] = 20.0 * max(0, np.cos(phase)) * gc
        body_u.step(forces)
    comparisons["Uniform\n(all same)"] = body_u.center_of_mass()[0] - start
    
    names = list(comparisons.keys())
    vals = list(comparisons.values())
    colors = ['#e74c3c', '#95a5a6', '#3498db']
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85,
                 edgecolor='white', linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
               f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Forward Displacement", fontsize=11)
    ax.set_title("Evolved vs Baselines", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "evolution_energy_alpha003.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

def main():
    best_genome, history, total_time = evolve()
    
    # Analyze spatial correlation
    distances, phase_diffs, corr, phases = analyze_correlation(best_genome)
    
    print(f"\n{'='*60}")
    print(f"  DISCOVERY ANALYSIS")
    print(f"{'='*60}")
    print(f"  Distance-Phase correlation: r = {corr:.4f}")
    if corr > 0.2:
        print(f"  ★ STRONG spatial correlation discovered!")
        print(f"    Evolution independently found that nearby particles")
        print(f"    should have similar phases (coordinated motion).")
    elif corr > 0.05:
        print(f"  ○ Weak spatial correlation detected.")
    else:
        print(f"  ✗ No spatial correlation. Evolution found a different strategy.")
    print(f"{'='*60}")
    
    # Visualize
    fig_path = visualize(best_genome, history, total_time)
    
    # Save results
    results = {
        "experiment": "Evolutionary Discovery with Energy Penalty",
        "energy_penalty_alpha": ENERGY_PENALTY_ALPHA,
        "population_size": POP_SIZE,
        "n_generations": N_GENERATIONS,
        "best_fitness": float(best_genome.fitness),
        "spatial_correlation": float(corr),
        "elapsed_min": round(total_time / 60, 1),
        "best_genome": best_genome.genes.tolist(),
        "fitness_history": {
            "best": history["best_fitness"],
            "avg": history["avg_fitness"],
        },
        "figure": fig_path,
    }
    log_path = os.path.join(RESULTS_DIR, "evolution_energy_alpha003_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
