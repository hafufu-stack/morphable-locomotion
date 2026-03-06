# Morphable Locomotion

**Emergent Role Differentiation in Morphable Soft-Body Robots: From Symmetry Locks to Muscle Synergy**

GPU-accelerated evolutionary framework for studying morphable soft-body robots that approach, combine, and locomote as merged entities.

📄 **Paper**: [Zenodo DOI](https://doi.org/10.5281/zenodo.18883399) (latest version)

## Key Findings (12 Principles)

### 1. Symmetry Locks Theorem
Symmetric bodies with shared controllers produce synchronized behavior (r=0.742). Functional differentiation **cannot** emerge without asymmetry.

### 2. Inertial Asymmetry Principle *(v2)*
Only **mass** asymmetry drives differentiation. Stiffness (r=0.579) and shape (r=0.735) asymmetries fail — because only F=ma creates different accelerations from identical neural outputs.

| Asymmetry Type | r(Fx) | Fitness | Differentiated? |
|---|---|---|---|
| Mass 10:1 | **-0.009** | +105 | ✅ Yes |
| Stiffness 10:1 | 0.579 | +169 | ❌ No |
| Shape (grid) | 0.735 | +171 | ❌ No |

### 3. Transient Differentiation *(v2)*
Role differentiation follows a **V-shaped trajectory** over 1000 generations — peaking at gen 25 (r=0.056) then partially reabsorbing to r≈0.58.

### 4. Reward Paradox
Multi-phase fitness functions can mask poor locomotion performance. Decomposed fitness tracking is essential.

### 5. Symmetry Optimality *(v2)*
When body masses are evolvable genes, evolution maintains symmetric masses (ratio 1.11:1).

### 6. The Sweet Spot *(v3)*
Mass ratio **3:1** maximizes Differentiation×Fitness (93.5). The r(Fx) curve is **non-monotonic** — dropping to r=0.398 at 3:1, recovering to r=0.736 at 7:1, then plunging again to r=0.165 at 10:1.

| Ratio | Fitness | r(Fx) | Diff×Fit |
|---|---|---|---|
| 1:1 | +172.5 | 0.851 | 25.8 |
| **3:1** | **+155.2** | **0.398** | **93.5** ⭐ |
| 10:1 | +100.4 | 0.165 | 83.9 |

### 7. Degrees of Freedom Trap *(v3)*
Adding per-particle mass control (body-wise softmax + EMA) causes a **256-point fitness collapse** (+175 → −81) that persists even under curriculum learning. Parallels biological reliance on low-dimensional motor synergies.

### 8. Environmental Differentiation *(v3)*
Asymmetric ground friction induces **complete role differentiation** (r=−0.032) in **symmetric bodies**, achieving fitness +181. Intelligence emerges at the body-environment boundary.

### 9. No-Cost Differentiation *(v4)* 🔥
Friction asymmetry induces differentiation **without fitness cost** — fitness and differentiation increase simultaneously (Diff×Fit = **171.0**, 1.8× mass Sweet Spot). Fundamentally different from mass asymmetry trade-off.

### 10. Asymmetry Interference *(v4)*
Combining body + environmental asymmetry produces **destructive interference** (+179 → +169). The optimal strategy is a single asymmetry source.

### 11. Muscle Synergy Principle *(v4)* 🏆
Compressing dynamic mass control from 200D to **1D center-of-mass shift** overcomes the DoF Trap, achieving the **all-time highest fitness (+210)** — 21% above baseline. Parallels biological muscle synergy patterns.

| Condition | Outputs | Fitness | Improvement |
|---|---|---|---|
| Fixed (3-out) | 3 | +173 | baseline |
| **Synergy (α=0.9)** | **4 (1D shift)** | **+210** 🏆 | **+21%** |
| Per-particle (DoF Trap) | 4 (200D) | −81 | −147% |

### 12. Adaptive Robustness *(v4)*
Controllers evolved under environmental reversal maintain near-optimal performance (+171), trading only 4% peak performance for environmental generality.

## System

- **Physics**: 400-particle spring-mass soft bodies (2 × 200 particles, Delaunay triangulation)
- **Controller**: Evolved neural network (7→32→3, 356 parameters)
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
└── perdim_verification.py         # v4: DoF Trap mechanism verification
figures/                           # Generated visualizations
results/                           # JSON experiment logs
reports/                           # Extension experiment reports
papers/                            # LaTeX paper source
```

## Quick Start

```bash
# Evolve a V2.1 combining robot (GPU required)
python src/evolve_combine.py

# Run asymmetric force analysis (3 conditions, ~6 min)
python src/asymmetric_force_analysis.py

# v2: Generalization experiments (~20 min)
python src/extension_experiments.py

# v3: Sweet Spot + DoF Trap + Swamp Test (~35 min)
python src/season3_experiments.py

# v4: Season 4 experiments (~25 min each)
python src/season4_experiments.py
python src/season4b_experiments.py
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
