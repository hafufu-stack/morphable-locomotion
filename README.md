# Morphable Locomotion

**Emergent Role Differentiation in Morphable Soft-Body Robots: From Symmetry Locks to Muscle Synergy**

GPU-accelerated evolutionary framework for studying morphable soft-body robots that approach, combine, and locomote as merged entities.

📄 **Paper**: [Zenodo DOI](https://doi.org/10.5281/zenodo.18883399) (latest version)

## Key Findings (15 Principles)

### 1. Symmetry Locks Theorem
Symmetric bodies with shared controllers produce synchronized behavior (r=0.742). Functional differentiation **cannot** emerge without asymmetry. Confirmed for N-body systems (v5).

### 2. Inertial Asymmetry Principle *(v2)*
Only **mass** asymmetry drives differentiation. Stiffness (r=0.579) and shape (r=0.735) asymmetries fail — because only F=ma creates different accelerations from identical neural outputs.

### 3. Transient Differentiation *(v2)*
Role differentiation follows a **V-shaped trajectory** over 1000 generations — peaking at gen 25 (r=0.056) then partially reabsorbing to r≈0.58.

### 4. Reward Paradox
Multi-phase fitness functions can mask poor locomotion performance. Decomposed fitness tracking is essential.

### 5. Symmetry Optimality *(v2)*
When body masses are evolvable genes, evolution maintains symmetric masses (ratio 1.11:1).

### 6. The Sweet Spot *(v3)*
Mass ratio **3:1** maximizes Differentiation×Fitness (93.5). The r(Fx) curve is **non-monotonic**.

### 7. Degrees of Freedom Trap *(v3)*
Adding per-particle mass control causes a **256-point fitness collapse** (+175 → −81) that persists even under curriculum learning.

### 8. Environmental Differentiation *(v3)*
Asymmetric ground friction induces **complete role differentiation** (r=−0.032) in **symmetric bodies** at fitness +181.

### 9. No-Cost Differentiation *(v4)* 🔥
Friction asymmetry induces differentiation **without fitness cost** (Diff×Fit = **171.0**, 1.8× mass Sweet Spot).

### 10. Asymmetry Interference *(v4)*
Combining body + environmental asymmetry produces **destructive interference** (+179 → +169).

### 11. Muscle Synergy Principle *(v4)* 🏆
Compressing dynamic mass control to **1D center-of-mass shift** overcomes the DoF Trap (+210, 21% above baseline).

### 12. Adaptive Robustness *(v4)*
Controllers evolved under environmental reversal maintain near-optimal performance (+171).

### 13. Synergy Dimension Optimality *(v5)* 🏆
The optimal number of synergy dimensions is **2–3**, achieving all-time highest fitness (**+216**). Performance degrades gracefully — even 50D stays positive (+160), unlike the catastrophic DoF Trap (−81).

| Dims | Fitness | Interpretation |
|---|---|---|
| 0D | +173 | Baseline (fixed) |
| **2–3D** | **+216** 🏆 | Sweet Spot |
| 50D | +160 | Graceful degradation |

### 14. Dynamic Resonance *(v5)*
Dynamic mass control (Muscle Synergy) **bypasses** the Asymmetry Interference Principle, maintaining +211 under environmental asymmetry (vs +169 for fixed asymmetry).

### 15. N-Body Selective Differentiation *(v5)* 🔥
In 3-body systems (600 particles), differentiation is **selective** — only mass-different pairs differentiate while equal-mass pairs remain synchronized.

| Mass Ratio | r(0-1) | r(1-2) | r(0-2) | Interpretation |
|---|---|---|---|---|
| 1:1:1 | 0.931 | 0.912 | 0.944 | All synchronized |
| **3:1:1** | **−0.034** | **0.835** | **−0.010** | **Body 0 only differentiates** |
| 5:1:5 | 0.722 | 0.730 | **0.996** | Mirror symmetry |

## System

- **Physics**: 400–600 particle spring-mass soft bodies (Delaunay triangulation)
- **Controller**: Evolved neural network (7→32→3/4/5, up to 356+ parameters)
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
└── season5b_experiments.py        # v5: 2D Synergy×Env, 3-Body Combination
figures/                           # Generated visualizations
results/                           # JSON experiment logs
papers/                            # LaTeX paper source
```

## Quick Start

```bash
# Evolve a V2.1 combining robot (GPU required)
python src/evolve_combine.py

# v3: Sweet Spot + DoF Trap + Swamp Test (~35 min)
python src/season3_experiments.py

# v4: Season 4 experiments (~25 min each)
python src/season4_experiments.py
python src/season4b_experiments.py

# v5: Synergy Dimension Sweep + Dynamic Resonance (~60 min)
python src/season5_experiments.py

# v5: 3-Body Combination (~30 min)
python src/season5b_experiments.py
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
