# Morphable Locomotion

**Emergent Role Differentiation in Morphable Soft-Body Robots: Inertial Asymmetry as the Unique Driver and the Transient Nature of Specialization**

GPU-accelerated evolutionary framework for studying morphable soft-body robots that approach, combine, and locomote as merged entities.

📄 **Paper**: [Zenodo DOI](https://doi.org/10.5281/zenodo.15004380) (v2)

## Key Findings (5 Principles)

### 1. Symmetry Locks Theorem
Symmetric bodies with shared controllers produce synchronized behavior (r=0.742). Functional differentiation **cannot** emerge without physical asymmetry.

### 2. Inertial Asymmetry Principle *(v2)*
Only **mass** asymmetry drives differentiation. Stiffness (r=0.579) and shape (r=0.735) asymmetries fail to break synchronization — because only F=ma creates different accelerations from identical neural outputs (Mapping Inequivalence).

| Asymmetry Type | r(Fx) | Fitness | Differentiated? |
|---|---|---|---|
| Mass 10:1 | **-0.009** | +105 | ✅ Yes |
| Stiffness 10:1 | 0.579 | +169 | ❌ No |
| Shape (grid) | 0.735 | +171 | ❌ No |
| Mass + Stiffness | **0.069** | +94 | ✅ Yes |

### 3. Transient Differentiation *(v2)*
Role differentiation follows a **V-shaped trajectory** over 1000 generations:

| Phase | Generation | r(Fx) | Fitness | Description |
|---|---|---|---|---|
| 1 | 0 | 0.716 | +52 | Random initialization |
| **2** | **25** | **0.056** | **+103** | **Rapid differentiation** |
| 3 | 50–75 | 0.66–0.72 | +107–118 | Re-synchronization |
| 4 | 200–1000 | ~0.58 | +120–128 | Optimization plateau |

### 4. Reward Paradox
Multi-phase fitness functions can create misleading optimization landscapes. Decomposed fitness tracking is essential for diagnosing reward hacking.

### 5. Symmetry Optimality *(v2)*
When body masses are evolvable genes, evolution maintains symmetric masses (ratio 1.11:1). Differentiation requires **externally imposed** asymmetry.

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
└── differentiation_dynamics.py    # v2: 1000-gen temporal tracking
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

# v2: Differentiation dynamics over 1000 generations (~20 min)
python src/differentiation_dynamics.py
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
