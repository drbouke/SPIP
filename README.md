# SPIP — Symbolic Path Inversion Problem

> A simulation and analysis toolkit for evaluating a structure-free, post-quantum cryptographic hardness assumption based on symbolic chaos.

This repository implements the simulation framework and analysis tools described in the paper:

"On the Intractability of Chaotic Symbolic Walks: Toward a Non-Algebraic Post-Quantum Hardness Assumption"  
by Mohamed Aly Bouke (2025)

---

## About SPIP

SPIP proposes a new computational hardness assumption based on chaotic symbolic dynamics over the discrete lattice Z^2. It avoids traditional algebraic structures and demonstrates exponential inversion complexity through:

- Symbolic trajectory explosion  
- Rounding-induced non-injectivity  
- Path collisions and entropy saturation  

This repository includes a full simulation pipeline to generate, visualize, and analyze symbolic trajectories under SPIP dynamics.

---

## Project Structure


SPIP/
├── spip_simulation.py       # Main simulation script
├── figures/                 # Auto-generated plots (after run)
├── results/                 # Summary statistics and CSV exports
└── README.md                # This file
