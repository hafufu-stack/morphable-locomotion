# Morphable Locomotion

**Emergent Role Differentiation in Morphable Soft-Body Robots: From Symmetry Locks to Muscle Synergy**

GPU-accelerated evolutionary framework for studying morphable soft-body robots that approach, combine, and locomote as merged entities.

📄 **Paper**: [Zenodo DOI](https://doi.org/10.5281/zenodo.18883399) (latest version)

## Key Findings (27 Principles)

### Theme I: Symmetry & Differentiation (1–5)
1. **Symmetry Locks Theorem** — Symmetric bodies synchronize (r=0.742). Differentiation requires asymmetry.
2. **Inertial Asymmetry Principle** — Only **mass** asymmetry drives differentiation (F=ma).
3. **Transient Differentiation** — V-shaped trajectory: peak at gen 25, partial reabsorption.
4. **Reward Paradox** — Multi-phase fitness can mask poor locomotion.
5. **Symmetry Optimality** — Evolution maintains symmetric masses (1.11:1).

### Theme II: Efficiency–Specialization Trade-offs (6–8)
6. **The Sweet Spot** — Mass ratio **3:1** maximizes Diff×Fit (93.5).
7. **Degrees of Freedom Trap** — Per-particle control collapses fitness (+175 → −81).
8. **Environmental Differentiation** — Asymmetric friction induces differentiation (r=−0.032) at +181.

### Theme III: Environmental Control (9–10)
9. **No-Cost Differentiation** 🔥 — Friction differentiates without fitness cost (Diff×Fit = 171.0).
10. **Asymmetry Interference** — Body + env asymmetry = destructive interference.

### Theme IV: Dimensional Compression (11–14)
11. **Muscle Synergy** 🏆 — 1D center-of-mass shift overcomes DoF Trap (+210).
12. **Adaptive Robustness** — Reversal-trained controllers maintain +171.
13. **Synergy Dimension Optimality** 🏆 — 2–3D synergy = all-time record **+216**.
14. **Dynamic Resonance** — Synergy bypasses Interference (+211 under env asymmetry).

### Theme V: Architectural Invariance (15–19)
15. **N-Body Selective Differentiation** 🔥 — Only mass-different pairs differentiate; mirror sync r=0.996.
16. **Neural Synchronization** 🧠 — Spring cutting doesn't disrupt mirror sync (r₀₂ ≥ 0.977). PCA confirms identical neural manifold trajectories (77.1% variance in 2 PCs).
17. **Convergent Strategy** — Independent NNs synchronize (r=0.952) without weight sharing.
18. **Physical Inevitability** 🔥 — Independent NNs differentiate **more strongly** (r=−0.301) than shared NNs (r=0.571).
19. **Parasitic Drag Cost** — Dead body costs 61% fitness. Cooperation = 2.6× solo.

### Theme VI: Social Dynamics (20–27)
20. **Parasitic Phase Transition** 🔥 — Co-evolutionary energy costs produce freeriders. Dose-response gap: 0 → +70 across α=0–10. Critical transition at α ≈ 2. Unconditional worker emerges (no reciprocal altruism).
21. **Architecture–Environment Interaction** — Independent NNs + friction = highest fitness (+189) but weaker differentiation (r=0.430).
22. **Topology Pulsation** 🤖 — Evolution discovers 31-cycle separation/recombination per simulation without fitness loss.
23. **Structural Constraint** — Opponent energy observation does not reduce freeriding (gap +19.3 vs blind +19.2). Parasitism is structural, not informational.
24. **Neural Manifold Geometry** 🧠 — Symmetry Locks = trajectory overlap in low-dimensional neural manifold. 1:1 → shared orbit (dist 1.51); 3:1 → separated (dist 2.40).
25. **Punishment Principle** 🔥 — Observation + physical disconnection ("divorce") enables autonomous punishment. Gap collapses from +12.9 to −2.1 with 10.8 divorces.
26. **Signal Irrelevance** 🆕 — Self-reported signals degenerate into Cheap Talk (signal–truth r ≈ 0.015). Divorce relies on physical spring tension, not self-reports.
27. **Physical Irreducibility** 🆕 — RNN controllers with temporal memory produce **larger** gaps (+20.1 vs FFN +16.5). Memory amplifies parasitism; the Structural Constraint is physically irreducible.

## System

- **Physics**: 400–600 particle spring-mass soft bodies (Delaunay triangulation)
- **Controller**: Evolved neural network (7–8→16–32→3–5, up to 712 parameters for dual-brain; RNN variant with persistent hidden state)
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
├── season6b_experiments.py        # v6: Physical Inevitability, Dead Body
├── season7_experiments.py         # v7: Parasite's Dilemma, Indep×Friction
├── exp21b_parasite_sweep.py       # v7: 7-point alpha dose-response
├── exp21c_parasite_mass.py        # v7: Parasite × Mass factorial
├── exp23_topology_control.py      # v7: Topology pulsation
├── stat_validation.py             # v8: Multi-seed statistical validation
├── exp24_reciprocal_altruism.py   # v9: Structural Constraint + PCA brain analysis
├── exp25_pca_comparison.py        # v10: Neural Manifold Geometry (PCA)
├── exp26_divorce.py               # v10: Evolution of Divorce (punishment)
├── exp27_deception.py             # v11: Evolution of Deception (signal irrelevance)
└── exp28_rnn_memory.py            # v11: RNN Memory (physical irreducibility)
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

# v7: Parasite's Dilemma + Topology Control (~90 min)
python src/season7_experiments.py
python src/exp21b_parasite_sweep.py
python src/exp23_topology_control.py

# v8: Statistical validation (~45 min)
python src/stat_validation.py

# v9–v11: Social dynamics deep dive (~4h total)
python src/exp24_reciprocal_altruism.py
python src/exp25_pca_comparison.py
python src/exp26_divorce.py
python src/exp27_deception.py
python src/exp28_rnn_memory.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA)
- NumPy, SciPy, Matplotlib, scikit-learn

## Author

Hiroto Funasaki — Independent Researcher, Japan  
ORCID: [0009-0004-2517-0177](https://orcid.org/0009-0004-2517-0177)

## License

MIT
