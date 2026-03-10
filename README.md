# Morphable Locomotion

**Emergent Role Differentiation in Morphable Soft-Body Robots: From Symmetry Locks to Muscle Synergy**

GPU-accelerated evolutionary framework for studying morphable soft-body robots that approach, combine, and locomote as merged entities.

📄 **Paper**: [Zenodo DOI](https://doi.org/10.5281/zenodo.18883399) (latest version)

## Key Findings (19 Principles)

### Theme I: Symmetry & Differentiation (1–5)
1. **Symmetry Locks Theorem** — Symmetric bodies synchronize (r=0.742). Differentiation requires asymmetry. Confirmed for N-body and decentralized architectures.
2. **Inertial Asymmetry Principle** *(v2)* — Only **mass** asymmetry drives differentiation (F=ma).
3. **Transient Differentiation** *(v2)* — V-shaped trajectory: peak at gen 25, partial reabsorption.
4. **Reward Paradox** — Multi-phase fitness can mask poor locomotion.
5. **Symmetry Optimality** *(v2)* — Evolution maintains symmetric masses (1.11:1).

### Theme II: Efficiency–Specialization Trade-offs (6–8)
6. **The Sweet Spot** *(v3)* — Mass ratio **3:1** maximizes Diff×Fit (93.5).
7. **Degrees of Freedom Trap** *(v3)* — Per-particle control collapses fitness (+175 → −81).
8. **Environmental Differentiation** *(v3)* — Asymmetric friction induces differentiation (r=−0.032) at +181.

### Theme III: Environmental Control (9–10)
9. **No-Cost Differentiation** *(v4)* 🔥 — Friction differentiates without fitness cost (Diff×Fit = 171.0).
10. **Asymmetry Interference** *(v4)* — Body + env asymmetry = destructive interference.

### Theme IV: Dimensional Compression (11–14)
11. **Muscle Synergy** *(v4)* 🏆 — 1D center-of-mass shift overcomes DoF Trap (+210).
12. **Adaptive Robustness** *(v4)* — Reversal-trained controllers maintain +171.
13. **Synergy Dimension Optimality** *(v5)* 🏆 — 2–3D synergy = all-time record **+216**.
14. **Dynamic Resonance** *(v5)* — Synergy bypasses Interference (+211 under env asymmetry).

### Theme V: Architectural Invariance (15–19) 🆕
15. **N-Body Selective Differentiation** *(v5)* 🔥 — Only mass-different pairs differentiate; mirror sync r=0.996.
16. **Neural Synchronization** *(v6)* 🧠 — Spring cutting doesn't disrupt mirror sync (r₀₂ ≥ 0.977). Sync is neural, not physical.
17. **Convergent Strategy** *(v6)* — Independent NNs synchronize (r=0.952) without weight sharing.
18. **Physical Inevitability** *(v6)* 🔥 — Independent NNs differentiate **more strongly** (r=−0.301) than shared NNs (r=0.571) under mass asymmetry.
19. **Parasitic Drag Cost** *(v6)* — Dead body costs 61% fitness. Cooperation = 2.6× solo.

| Architecture | Mass | Fitness | r(Fx) |
|---|---|---|---|
| Shared NN | 1:1 | +174 | 0.942 |
| Independent NNs | 1:1 | +167 | 0.952 |
| Shared NN | 3:1 | +100 | 0.571 |
| **Independent NNs** | **3:1** | **+96** | **−0.301** 🔥 |

## System

- **Physics**: 400–600 particle spring-mass soft bodies (Delaunay triangulation)
- **Controller**: Evolved neural network (7→32→3/4/5, up to 712 parameters for dual-brain)
- **Evolution**: (μ+λ) strategy, 200 population, up to 1000 generations
- **GPU**: Full batch parallelization on NVIDIA RTX 5080 Laptop (< 3 min per 150-gen run)
- **Framework**: PyTorch (CUDA)

## Project Structure

```
src/
├── evolve_combine.py              # V2.1 combining robot evolution
├── evolve_nn.py                   # Single-body NN evolution
├── triple_batch.py                # Bone & Muscle, Dual-Mode batch
├── paradox_investigation.py       # Reward paradox decomposition
├── force_analysis.py              # Per-body force tracking
├── asymmetric_force_analysis.py   # Mass asymmetry differentiation
├── extension_experiments.py       # v2: Generalization (stiffness/shape/combined)
├── differentiation_dynamics.py    # v2: 1000-gen temporal tracking
├── dynamic_mass_transfer.py       # v3: Dynamic mass transfer (DoF Trap)
├── season3_experiments.py         # v3: Sweet Spot, Dev Unlocking, Swamp Test
├── season4_experiments.py         # v4: Friction Sweet Spot, Double Asymmetry
├── season4b_experiments.py        # v4: Muscle Synergy, Environmental Reversal
├── perdim_verification.py         # v4: DoF Trap mechanism verification
├── season5_experiments.py         # v5: Synergy Dimension Sweep, Dynamic Resonance
├── season5b_experiments.py        # v5: 2D Synergy×Env, 3-Body Combination
├── season6_experiments.py         # v6: Phantom Synchronization, Decentralized Brains
└── season6b_experiments.py        # v6: Physical Inevitability, Dead Body
figures/                           # Generated visualizations
results/                           # JSON experiment logs
papers/                            # LaTeX paper source
```

## Quick Start

```bash
# v3: Sweet Spot + DoF Trap (~35 min)
python src/season3_experiments.py

# v4: Season 4 experiments (~25 min each)
python src/season4_experiments.py
python src/season4b_experiments.py

# v5: Synergy Dimension + 3-Body (~90 min)
python src/season5_experiments.py
python src/season5b_experiments.py

# v6: Phantom Sync + Decentralized + Dead Body (~40 min)
python src/season6_experiments.py
python src/season6b_experiments.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA)
- NumPy, SciPy, Matplotlib

## Author

Hiroto Funasaki — Independent Researcher, Japan  
ORCID: [0009-0004-2517-0177](https://orcid.org/0009-0004-2517-0177)

## License

MIT
