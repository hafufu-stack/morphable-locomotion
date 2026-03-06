"""
Extended CPG: Beat the Handcoded Optimal (68.79)!
=================================================

Extends the 5-gene CPG to ~12 genes with:
- Dual-frequency oscillation (fundamental + harmonic)
- Independent X/Y amplitude control per frequency
- Left-right asymmetry gene
- Height-dependent activation (top vs bottom particles)

Target: Beat handcoded "Optimal" score of 68.79

Usage:
    python src/evolve_cpg_extended.py
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
POP_SIZE = 300          # Large pop for 12D search
N_GENERATIONS = 200     # Thorough search
MUTATION_RATE = 0.25
MUTATION_SIGMA = 0.15
ELITE_FRAC = 0.1
TOURNAMENT_SIZE = 4

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


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
#  EXTENDED CPG GENOME: 12 parameters
# ===================================================

class ExtCPGGenome:
    """Extended CPG with dual frequencies and asymmetry.
    
    genes[0]  = freq1      : primary frequency (Hz)
    genes[1]  = amp_x1     : primary X amplitude
    genes[2]  = amp_y1     : primary Y amplitude
    genes[3]  = k_x1       : primary X phase gradient
    genes[4]  = k_y1       : primary Y phase gradient
    genes[5]  = freq2      : secondary frequency (harmonic)
    genes[6]  = amp_x2     : secondary X amplitude
    genes[7]  = amp_y2     : secondary Y amplitude
    genes[8]  = k_x2       : secondary X phase gradient
    genes[9]  = k_y2       : secondary Y phase gradient
    genes[10] = asym       : left-right asymmetry factor
    genes[11] = height_mod : height-dependent modulation
    """
    N_GENES = 12
    
    # Per-gene initialization ranges
    INIT_RANGES = [
        (0.3, 3.0),    # freq1
        (-2.0, 2.0),   # amp_x1
        (-2.0, 2.0),   # amp_y1
        (-5.0, 5.0),   # k_x1
        (-5.0, 5.0),   # k_y1
        (0.5, 6.0),    # freq2 (often harmonic of freq1)
        (-1.5, 1.5),   # amp_x2
        (-1.5, 1.5),   # amp_y2
        (-5.0, 5.0),   # k_x2
        (-5.0, 5.0),   # k_y2
        (-1.0, 1.0),   # asym
        (-1.0, 1.0),   # height_mod
    ]
    
    # Mutation scale per gene
    MUT_SCALES = [0.2, 0.3, 0.3, 0.5, 0.5, 0.3, 0.3, 0.3, 0.5, 0.5, 0.2, 0.2]
    
    def __init__(self):
        self.genes = np.array([np.random.uniform(lo, hi) for lo, hi in self.INIT_RANGES])
        self.fitness = -np.inf

    def copy(self):
        g = ExtCPGGenome.__new__(ExtCPGGenome)
        g.genes = self.genes.copy()
        g.fitness = self.fitness
        return g

    def mutate(self, rate=MUTATION_RATE, sigma=MUTATION_SIGMA):
        for i in range(self.N_GENES):
            if np.random.random() < rate:
                self.genes[i] += np.random.randn() * sigma * self.MUT_SCALES[i]

    def generate_forces(self, body, t):
        BASE_AMP = 40.0
        n = body.n
        forces = np.zeros((n, 2))
        
        freq1 = abs(self.genes[0])
        amp_x1 = BASE_AMP * np.tanh(self.genes[1])
        amp_y1 = BASE_AMP * np.tanh(self.genes[2])
        k_x1 = self.genes[3]
        k_y1 = self.genes[4]
        
        freq2 = abs(self.genes[5])
        amp_x2 = BASE_AMP * np.tanh(self.genes[6]) * 0.5  # Secondary is weaker
        amp_y2 = BASE_AMP * np.tanh(self.genes[7]) * 0.5
        k_x2 = self.genes[8]
        k_y2 = self.genes[9]
        
        asym = self.genes[10]
        height_mod = self.genes[11]
        
        # Precompute rest positions info
        y_min = body.rest_pos[:, 1].min()
        y_max = body.rest_pos[:, 1].max()
        y_range = y_max - y_min + 1e-8
        x_center = np.mean(body.rest_pos[:, 0])
        
        for i in range(n):
            x_i = body.rest_pos[i, 0]
            y_i = body.rest_pos[i, 1]
            
            # Primary CPG wave
            phase1 = 2 * np.pi * freq1 * t + k_x1 * x_i + k_y1 * y_i
            fx1 = amp_x1 * np.sin(phase1)
            fy1 = amp_y1 * max(0, np.cos(phase1))
            
            # Secondary harmonic wave
            phase2 = 2 * np.pi * freq2 * t + k_x2 * x_i + k_y2 * y_i
            fx2 = amp_x2 * np.sin(phase2)
            fy2 = amp_y2 * max(0, np.cos(phase2))
            
            # Left-right asymmetry: particles on one side push harder
            side = np.tanh((x_i - x_center) * 2.0)  # -1 to +1
            asym_factor = 1.0 + asym * side
            
            # Height modulation: bottom particles behave differently
            h_norm = (y_i - y_min) / y_range  # 0=bottom, 1=top
            h_factor = 1.0 + height_mod * (1.0 - 2.0 * h_norm)  # bottom>top if height_mod>0
            
            # Ground contact boost
            ground_contact = 1.5 if body.pos[i, 1] < GROUND_Y + 0.3 else 0.5
            
            # Combined force
            total_fx = (fx1 + fx2) * asym_factor * h_factor * ground_contact
            total_fy = (fy1 + fy2) * h_factor * ground_contact
            
            forces[i, 0] = total_fx
            forces[i, 1] = total_fy
        
        return forces


def crossover(p1, p2):
    child = p1.copy()
    # Blend crossover (more effective for continuous params)
    alpha = np.random.uniform(0.0, 1.0, child.genes.shape)
    child.genes = alpha * p1.genes + (1 - alpha) * p2.genes
    child.fitness = -np.inf
    return child


def evaluate_fitness_ext(genes_array):
    g = ExtCPGGenome.__new__(ExtCPGGenome)
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


def tournament_select(population, k=TOURNAMENT_SIZE):
    candidates = np.random.choice(len(population), k, replace=False)
    best = max(candidates, key=lambda i: population[i].fitness)
    return population[best]


def evolve():
    print("="*60)
    print("  Extended CPG: Beat the Optimal (68.79)!")
    print(f"  Population: {POP_SIZE}, Generations: {N_GENERATIONS}")
    print(f"  Genome: {ExtCPGGenome.N_GENES} genes (dual-freq + asymmetry)")
    print(f"  Target: > 68.79 (handcoded optimal)")
    print(f"  Workers: {N_WORKERS} cores")
    print("="*60)
    
    t0 = time.time()
    population = [ExtCPGGenome() for _ in range(POP_SIZE)]
    
    history = {"best_fitness": [], "avg_fitness": [], "worst_fitness": [],
               "best_genes": []}
    
    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))
    best_ever = -np.inf
    target_reached = False
    
    for gen in range(N_GENERATIONS):
        gen_t0 = time.time()
        
        unevaluated = [(i, g) for i, g in enumerate(population) if g.fitness == -np.inf]
        if unevaluated:
            indices, genomes = zip(*unevaluated)
            genes_list = [g.genes.copy() for g in genomes]
            with Pool(N_WORKERS) as pool:
                fitnesses = pool.map(evaluate_fitness_ext, genes_list)
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
        
        if best.fitness > best_ever:
            best_ever = best.fitness
            marker = ""
            if best.fitness > 35.13:
                marker = " << BEAT CPG-5D!"
            if best.fitness > 68.79:
                marker = " << !!!! BEAT OPTIMAL !!!!"
                if not target_reached:
                    target_reached = True
                    print(f"\n  *** TARGET REACHED at Gen {gen}! ***\n")
            print(f"  Gen {gen:3d}/{N_GENERATIONS}: best={best.fitness:+8.2f}  "
                  f"avg={avg_fit:+8.2f}  [{gen_dt:.1f}s]{marker}")
        elif gen % 20 == 0:
            print(f"  Gen {gen:3d}/{N_GENERATIONS}: best={best.fitness:+8.2f}  "
                  f"avg={avg_fit:+8.2f}  [{gen_dt:.1f}s]")
        
        new_pop = [population[i].copy() for i in range(n_elite)]
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            child = crossover(p1, p2)
            child.mutate()
            new_pop.append(child)
        population = new_pop
    
    for g in population:
        if g.fitness == -np.inf:
            g.fitness = evaluate_fitness_ext(g.genes)
    population.sort(key=lambda g: g.fitness, reverse=True)
    
    total_time = time.time() - t0
    print(f"\n  Total time: {total_time/60:.1f} min")
    print(f"  Best fitness: {population[0].fitness:+.2f}")
    
    return population[0], history, total_time


def visualize(best, history, total_time):
    body = SoftBody()
    genes = best.genes
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    beat_optimal = best.fitness > 68.79
    title_color = 'green' if beat_optimal else 'black'
    result = "VICTORY!" if beat_optimal else f"Score: {best.fitness:.1f} / 68.79"
    
    fig.suptitle(f"Extended CPG (12 genes): {result}\n"
                 f"freq1={genes[0]:.2f} freq2={genes[5]:.2f} "
                 f"asym={genes[10]:.2f} h_mod={genes[11]:.2f}",
                 fontsize=13, fontweight="bold", color=title_color)
    
    # Panel 1: Fitness evolution with target line
    ax = axes[0][0]
    gens = range(len(history["best_fitness"]))
    ax.fill_between(gens, history["worst_fitness"], history["best_fitness"],
                   alpha=0.2, color='#3498db')
    ax.plot(gens, history["best_fitness"], 'b-', linewidth=2, label='Best')
    ax.plot(gens, history["avg_fitness"], 'g--', linewidth=1.5, label='Average')
    ax.axhline(y=68.79, color='red', linestyle='--', linewidth=2, label='Optimal (68.79)')
    ax.axhline(y=35.13, color='orange', linestyle=':', linewidth=1.5, label='CPG-5D (35.13)')
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness")
    ax.set_title("Can AI Beat Human Design?", fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 2: Phase map (primary wave)
    ax = axes[0][1]
    positions = body.rest_pos
    k_x1, k_y1 = genes[3], genes[4]
    phases = (k_x1 * positions[:, 0] + k_y1 * positions[:, 1]) % (2*np.pi)
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=phases,
                        cmap='hsv', s=200, edgecolors='black', linewidth=1,
                        vmin=0, vmax=2*np.pi)
    plt.colorbar(scatter, ax=ax, label='Primary Phase (rad)')
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Primary Wave Phase Map", fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    
    # Panel 3: Comparison bar chart
    ax = axes[1][0]
    comparisons = {
        "Independent\n(seizure)": 1.74,
        "CPG 5D\n(prev)": 35.13,
        "Extended\nCPG 12D": best.fitness,
        "Handcoded\nOptimal": 68.79,
    }
    names = list(comparisons.keys())
    vals = list(comparisons.values())
    colors = ['#95a5a6', '#f39c12', '#e74c3c', '#3498db']
    if best.fitness > 68.79:
        colors[2] = '#27ae60'  # Green if beat optimal
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85,
                 edgecolor='white', linewidth=2)
    for b, v in zip(bars, vals):
        y = max(v, 0) + 1
        ax.text(b.get_x()+b.get_width()/2, y, f'{v:.1f}',
               ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Forward Displacement")
    ax.set_title("Evolution vs Human Design", fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Panel 4: Gene values radar-ish display
    ax = axes[1][1]
    gene_names = ['freq1', 'amp_x1', 'amp_y1', 'k_x1', 'k_y1',
                  'freq2', 'amp_x2', 'amp_y2', 'k_x2', 'k_y2',
                  'asym', 'h_mod']
    gene_vals = genes
    colors_g = plt.cm.Spectral(np.linspace(0, 1, len(gene_names)))
    bars = ax.barh(range(len(gene_names)), gene_vals, color=colors_g, alpha=0.8)
    ax.set_yticks(range(len(gene_names)))
    ax.set_yticklabels(gene_names, fontsize=9)
    ax.set_xlabel("Gene Value")
    ax.set_title("Evolved Gene Values (12D)", fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "evolution_cpg_extended.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {path}")
    return path


def main():
    best, history, total_time = evolve()
    genes = best.genes
    
    print(f"\n{'='*60}")
    print(f"  EXTENDED CPG RESULTS")
    print(f"{'='*60}")
    print(f"  Best fitness: {best.fitness:.2f}")
    print(f"  Target (Optimal): 68.79")
    print(f"  {'VICTORY! AI beats human design!' if best.fitness > 68.79 else 'Not quite... but impressive!'}")
    print(f"")
    print(f"  Primary wave:   freq={genes[0]:.3f}Hz  k=({genes[3]:.2f}, {genes[4]:.2f})")
    print(f"  Secondary wave: freq={genes[5]:.3f}Hz  k=({genes[8]:.2f}, {genes[9]:.2f})")
    print(f"  Asymmetry: {genes[10]:.3f}  Height mod: {genes[11]:.3f}")
    print(f"{'='*60}")
    
    fig_path = visualize(best, history, total_time)
    
    results = {
        "experiment": "Extended CPG vs Handcoded Optimal",
        "genome_size": ExtCPGGenome.N_GENES,
        "target_fitness": 68.79,
        "best_fitness": float(best.fitness),
        "beat_target": bool(best.fitness > 68.79),
        "best_genes": {n: float(v) for n, v in zip(
            ['freq1','amp_x1','amp_y1','k_x1','k_y1',
             'freq2','amp_x2','amp_y2','k_x2','k_y2',
             'asym','height_mod'], genes)},
        "population_size": POP_SIZE,
        "n_generations": N_GENERATIONS,
        "elapsed_min": round(total_time / 60, 1),
        "fitness_history": {
            "best": history["best_fitness"],
            "avg": history["avg_fitness"],
        },
        "figure": fig_path,
    }
    log_path = os.path.join(RESULTS_DIR, "evolution_cpg_extended_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
