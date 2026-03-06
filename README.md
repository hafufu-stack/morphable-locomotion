# Morphable Locomotion

**Emergent Role Differentiation in Morphable Soft-Body Robots: Mass Asymmetry as a Necessary Condition for Functional Specialization**

GPU-accelerated evolutionary framework for studying morphable soft-body robots that approach, combine, and locomote as merged entities.

## Key Findings

| Condition | Mass Ratio | r(Fx) | Fitness | Status |
|-----------|-----------|-------|---------|--------|
| Symmetric | 1:1 | 0.742 | +166 | Synchronized |
| Bone+Muscle | 4:1 | 0.368 | +145 | Weakly correlated |
| Extreme | 10:1 | **0.020** | +105 | **Differentiated** |

1. **Symmetry Locks Theorem**: Symmetric bodies with shared controllers produce synchronized behavior. Functional differentiation cannot emerge without physical asymmetry.
2. **Mass Asymmetry Differentiation Principle**: A 10:1 mass ratio drives force correlation to near-zero (r=0.020), demonstrating that structural heterogeneity induces emergent role specialization.
3. **Reward Paradox**: Multi-phase fitness functions can create misleading optimization landscapes. Decomposed fitness tracking is essential for diagnosing reward hacking.

## System

- **Physics**: 400-particle spring-mass soft bodies (2 × 200 particles, Delaunay triangulation)
- **Controller**: Evolved neural network (7→32→3, 356 parameters)
- **Evolution**: (μ+λ) strategy, 200 population × 150 generations
- **GPU**: Full batch parallelization on NVIDIA RTX 5080 Laptop (< 3 min per run)
- **Framework**: PyTorch (CUDA)

## Project Structure

```
src/                    # Experiment scripts
├── evolve_combine.py   # V2.1 combining robot evolution
├── evolve_nn.py        # Single-body NN evolution
├── triple_batch.py     # Bone & Muscle, Dual-Mode batch
├── paradox_investigation.py   # Reward paradox decomposition
├── force_analysis.py          # Per-body force tracking
└── asymmetric_force_analysis.py  # Mass asymmetry differentiation
figures/                # Generated visualizations
results/                # JSON experiment logs
papers/                 # LaTeX paper (not tracked in git)
```

## Quick Start

```bash
# Evolve a V2.1 combining robot (GPU required)
python src/evolve_combine.py

# Run asymmetric force analysis (3 conditions, ~6 min)
python src/asymmetric_force_analysis.py
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
