"""
Morphable Soft-Body Locomotion Simulation
==========================================

CPU-only 2D physics simulation demonstrating WHY correlated signals
matter for physical body control, contrasting with LLM reasoning
(where correlation doesn't help due to single residual stream).

The "body" is a deformable 2D mesh of particles connected by springs.
A neural controller outputs forces for each particle.
We compare locomotion under different signal correlation modes:

  1. Correlated (ﾏ≫沿+1 nearby, ﾏ≫沿-1 distant) 竊・Smooth coordinated walk
  2. Independent (ﾏ・0) 竊・Chaotic seizure
  3. Anti-correlated (ﾏ・-1 everywhere) 竊・Rigid oscillation
  4. Optimal (distance-based ﾏ・ 竊・Best locomotion

Output: Animated GIF of all 4 modes side by side.

Usage:
    python src/morphable_locomotion.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.spatial import Delaunay
import os, time

# === Config ===
N_PARTICLES = 36  # 6x6 grid
GRID_SIZE = 6
DT = 0.015         # Time step
N_STEPS = 600      # Total simulation steps (longer for clearer locomotion)
SPRING_K = 20.0    # Softer springs for more deformable body
SPRING_DAMP = 1.0  # Less damping
DRAG = 0.3         # Less drag
SIGNAL_FREQ = 1.5  # Hz (slower gait)
SIGNAL_AMP = 40.0  # Stronger forces for visible movement
GROUND_Y = -0.5    # Ground level
GROUND_K = 500.0   # Stronger ground repulsion (bouncy)
GRAVITY = -8.0     # Stronger gravity for ground contact

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================
#  BODY: Deformable mesh of spring-connected particles
# ===================================================

class SoftBody:
    """2D deformable body: a grid of particles connected by springs."""

    def __init__(self, cx=0, cy=2.0):
        # Initialize particles in a grid
        self.n = N_PARTICLES
        self.pos = np.zeros((self.n, 2))
        self.vel = np.zeros((self.n, 2))
        self.mass = np.ones(self.n) * 1.0

        # Create grid positions
        idx = 0
        spacing = 0.5
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                self.pos[idx] = [cx + col * spacing - (GRID_SIZE-1)*spacing/2,
                                 cy + row * spacing]
                idx += 1

        # Rest positions (for spring rest lengths)
        self.rest_pos = self.pos.copy()

        # Create springs via Delaunay triangulation
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

        # Particle distances (for correlation computation)
        self.distances = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.distances[i, j] = np.linalg.norm(self.rest_pos[i] - self.rest_pos[j])
        self.max_dist = self.distances.max()

    def step(self, external_forces):
        """Physics step with spring forces, gravity, ground, drag, and external forces."""
        forces = np.zeros_like(self.pos)

        # Gravity
        forces[:, 1] += GRAVITY * self.mass

        # Spring forces
        for a, b in self.springs:
            diff = self.pos[b] - self.pos[a]
            dist = np.linalg.norm(diff) + 1e-8
            direction = diff / dist
            rest = self.rest_lengths[(a, b)]
            stretch = dist - rest
            
            # Spring + damping
            rel_vel = self.vel[b] - self.vel[a]
            f_spring = SPRING_K * stretch * direction
            f_damp = SPRING_DAMP * np.dot(rel_vel, direction) * direction
            
            forces[a] += f_spring + f_damp
            forces[b] -= f_spring + f_damp

        # Ground collision
        for i in range(self.n):
            if self.pos[i, 1] < GROUND_Y:
                penetration = GROUND_Y - self.pos[i, 1]
                forces[i, 1] += GROUND_K * penetration
                # Friction
                forces[i, 0] -= 3.0 * self.vel[i, 0]

        # Drag
        forces -= DRAG * self.vel

        # External (neural) forces
        forces += external_forces

        # Euler integration
        acc = forces / self.mass[:, np.newaxis]
        self.vel += acc * DT
        self.pos += self.vel * DT

    def center_of_mass(self):
        return np.mean(self.pos, axis=0)

    def body_extent(self):
        return np.max(self.pos, axis=0) - np.min(self.pos, axis=0)


# ===================================================
#  NEURAL CONTROLLERS (different correlation modes)
# ===================================================

def generate_correlated_forces(body, t, mode="correlated"):
    """Generate neural control signals for each particle.
    
    Each particle gets a 2D force vector from a neural oscillator.
    The key difference is HOW the oscillator phases are correlated
    across particles.
    """
    n = body.n
    forces = np.zeros((n, 2))

    if mode == "correlated":
        # Traveling wave across body: nearby particles move in sync
        # This creates caterpillar-like locomotion
        for i in range(n):
            x_pos = body.rest_pos[i, 0]
            y_pos = body.rest_pos[i, 1]
            
            y_min = body.rest_pos[:, 1].min()
            y_max = body.rest_pos[:, 1].max()
            leg_factor = max(0, 1.0 - (y_pos - y_min) / (y_max - y_min + 1e-8))

            # Traveling wave with wavelength matching body width
            phase = 2 * np.pi * SIGNAL_FREQ * t + x_pos * 4.0
            
            # Strong forward push on ground contact, lift to clear obstacle
            ground_contact = 1.0 if body.pos[i, 1] < GROUND_Y + 0.3 else 0.3
            forces[i, 0] = SIGNAL_AMP * np.sin(phase) * ground_contact
            forces[i, 1] = SIGNAL_AMP * 0.6 * max(0, np.cos(phase)) * leg_factor

    elif mode == "independent":
        # Each particle gets INDEPENDENT random oscillation (seizure!)
        # Use deterministic per-particle phases for reproducibility
        rng = np.random.RandomState(42)
        phases = rng.uniform(0, 2*np.pi, n)
        freqs = rng.uniform(0.5, 4.0, n)
        for i in range(n):
            forces[i, 0] = SIGNAL_AMP * np.sin(2*np.pi * freqs[i] * t + phases[i])
            forces[i, 1] = SIGNAL_AMP * 0.5 * np.sin(2*np.pi * freqs[i] * 1.3 * t + phases[i] + 1.0)

    elif mode == "anti_correlated":
        # ALL particles are anti-correlated: even particles go +, odd go -
        for i in range(n):
            sign = 1 if i % 2 == 0 else -1
            phase = 2 * np.pi * SIGNAL_FREQ * t
            forces[i, 0] = SIGNAL_AMP * sign * np.sin(phase)
            forces[i, 1] = SIGNAL_AMP * 0.3 * sign * np.cos(phase)

    elif mode == "optimal":
        # Structured gait: left-right anti-correlation + vertical-based role
        # Bottom = legs (push), Top = body (stabilize)
        x_center = np.mean(body.rest_pos[:, 0])
        y_min = body.rest_pos[:, 1].min()
        y_max = body.rest_pos[:, 1].max()
        
        for i in range(n):
            x_pos = body.rest_pos[i, 0]
            y_pos = body.rest_pos[i, 1]
            
            # Left-right anti-correlation (walking gait)
            side = 1 if x_pos >= x_center else -1
            height_ratio = (y_pos - y_min) / (y_max - y_min + 1e-8)
            leg_factor = max(0, 1.0 - height_ratio)
            
            # Gait: alternating left-right push with vertical lift
            phase = 2 * np.pi * SIGNAL_FREQ * t + side * np.pi
            ground_contact = 1.0 if body.pos[i, 1] < GROUND_Y + 0.3 else 0.3
            
            # Forward bias + gait oscillation
            forces[i, 0] = SIGNAL_AMP * (0.3 + 0.7 * np.sin(phase) * ground_contact) * leg_factor
            forces[i, 1] = SIGNAL_AMP * 0.5 * max(0, np.cos(phase)) * leg_factor
            
            # Upper body: stabilization force toward COM
            if height_ratio > 0.5:
                forces[i, 0] *= 0.3  # Upper body doesn't push as hard
                forces[i, 1] += SIGNAL_AMP * 0.1  # Slight upward to maintain posture

    return forces


# ===================================================
#  SIMULATION
# ===================================================

def run_simulation(mode, seed=42):
    """Run one simulation and return trajectory."""
    np.random.seed(seed)
    body = SoftBody(cx=0, cy=2.0)

    trajectory = []
    com_history = []

    for step in range(N_STEPS):
        t = step * DT
        forces = generate_correlated_forces(body, t, mode)
        body.step(forces)
        
        if step % 2 == 0:  # Save every other frame
            trajectory.append(body.pos.copy())
            com_history.append(body.center_of_mass().copy())

    return trajectory, com_history, body


def main():
    print("="*60)
    print("  Morphable Soft-Body Locomotion Simulation")
    print("  CPU-only 2D Physics")
    print("="*60)

    modes = ["correlated", "independent", "anti_correlated", "optimal"]
    labels = [
        "Correlated (ﾏ・wave)\n'Smooth Walk'",
        "Independent (ﾏ・0)\n'Neural Seizure'",
        "Anti-correlated\n'Rigid Oscillation'",
        "Optimal (Structured)\n'Distance-based ﾏ・"
    ]

    # Run all simulations
    results = {}
    for mode in modes:
        print(f"\n  Running {mode}...")
        t0 = time.time()
        traj, com, body = run_simulation(mode)
        dt = time.time() - t0
        
        displacement = com[-1][0] - com[0][0]
        print(f"    {len(traj)} frames, {dt:.1f}s")
        print(f"    Net displacement: {displacement:.2f} units")
        results[mode] = {"traj": traj, "com": com, "displacement": displacement}

    # === Create animated GIF ===
    print("\n  Creating animation...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Morphable Body: Signal Correlation vs Locomotion\n"
                 "Nearby particles need CORRELATED signals for coordinated motion",
                 fontsize=13, fontweight="bold")

    # Pre-compute axis limits
    all_x, all_y = [], []
    for mode in modes:
        for frame in results[mode]["traj"]:
            all_x.extend(frame[:, 0])
            all_y.extend(frame[:, 1])
    x_min, x_max = min(all_x) - 1, max(all_x) + 1
    y_min = GROUND_Y - 0.5
    y_max = max(all_y) + 1

    # Setup axes
    scatters = []
    com_lines = []
    displacement_texts = []
    for idx, (mode, label) in enumerate(zip(modes, labels)):
        ax = axes[idx // 2][idx % 2]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.axhline(y=GROUND_Y, color='#8B4513', linewidth=3, alpha=0.6)
        ax.fill_between([x_min, x_max], [y_min, y_min], [GROUND_Y, GROUND_Y],
                       color='#D2691E', alpha=0.15)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")

        # Initialize scatter
        s = ax.scatter([], [], c='#C0C0C0', s=60, edgecolors='#404040',
                      linewidth=0.5, zorder=5)
        scatters.append(s)

        # COM trail
        line, = ax.plot([], [], 'r-', alpha=0.4, linewidth=1)
        com_lines.append(line)

        # Displacement text
        txt = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                     fontsize=9, fontweight='bold', va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        displacement_texts.append(txt)

    def animate(frame_idx):
        for idx, mode in enumerate(modes):
            traj = results[mode]["traj"]
            com = results[mode]["com"]
            
            if frame_idx < len(traj):
                pos = traj[frame_idx]
                scatters[idx].set_offsets(pos)

                # Color particles by velocity or role
                y_vals = pos[:, 1]
                colors = plt.cm.coolwarm((y_vals - y_vals.min()) / (y_vals.max() - y_vals.min() + 1e-8))
                scatters[idx].set_color(colors)

                # COM trail
                trail = np.array(com[:frame_idx+1])
                com_lines[idx].set_data(trail[:, 0], trail[:, 1])

                # Displacement
                dx = com[frame_idx][0] - com[0][0]
                t = frame_idx * 2 * DT
                displacement_texts[idx].set_text(f'ﾎ肺={dx:.2f}  t={t:.1f}s')

        return scatters + com_lines + displacement_texts

    n_frames = len(results[modes[0]]["traj"])
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=50, blit=False)

    # Save as GIF
    gif_path = os.path.join(OUTPUT_DIR, "morphable_locomotion.gif")
    print(f"  Saving GIF to {gif_path}...")
    anim.save(gif_path, writer='pillow', fps=20)
    plt.close()

    # === Also save a static comparison figure ===
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Morphable Body Simulation: Correlation Drives Locomotion",
                  fontsize=14, fontweight="bold")

    # Panel 1: Final displacement
    ax = axes2[0]
    disps = [results[m]["displacement"] for m in modes]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    bars = ax.bar(range(4), disps, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Correlated\nWave", "Independent\nSeizure",
                        "Anti-corr\nRigid", "Optimal\\nStructured"], fontsize=9)
    ax.set_ylabel("Net X Displacement", fontsize=12)
    ax.set_title("Locomotion Distance by Correlation Mode", fontsize=12, fontweight='bold')
    for b, d in zip(bars, disps):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
               f'{d:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: COM trajectories
    ax = axes2[1]
    for idx, (mode, color) in enumerate(zip(modes, colors)):
        com = np.array(results[mode]["com"])
        ax.plot(com[:, 0], com[:, 1], color=color, linewidth=2,
               label=mode, alpha=0.8)
    ax.axhline(y=GROUND_Y, color='#8B4513', linewidth=2, alpha=0.4)
    ax.set_xlabel("X position", fontsize=12)
    ax.set_ylabel("Y position", fontsize=12)
    ax.set_title("Center of Mass Trajectory", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    static_path = os.path.join(OUTPUT_DIR, "morphable_comparison.png")
    plt.savefig(static_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Static figure: {static_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for mode, label in zip(modes, labels):
        d = results[mode]["displacement"]
        print(f"  {mode:20s}  ﾎ肺 = {d:+.2f}")
    print(f"\n  Analogy to SNN-Genesis:")
    print(f"  - LLM (Hanoi): Single residual stream 竊・correlation doesn't help")
    print(f"  - Physical body (soft-body): Many independent DOF 竊・correlation is ESSENTIAL")
    print(f"{'='*60}")
    print(f"\n  Files:")
    print(f"    GIF:    {gif_path}")
    print(f"    Static: {static_path}")


if __name__ == "__main__":
    main()

