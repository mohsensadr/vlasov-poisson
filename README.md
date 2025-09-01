![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Vlasov-Poisson with Variance Reduction

A CUDA/C++ implementation of the **particle-based solution** to the **Vlasov–Poisson equation** with **Variance Reduction** using **importance sampling weights**. This project provides a high-performance GPU-accelerated framework for simulating plasma dynamics with reduced statistical noise, enabling more accurate long-time evolution of distribution functions.

---

## Overview

The **Vlasov–Poisson equation** describes the evolution of a plasma or charged particle system under self-consistent electric fields. Traditional particle-in-cell (PIC) methods can suffer from high variance due to noisy sampling of velocity space. This implementation addresses that challenge by applying **variance reduction** through a **control variate** and dynamic **importance weights**. For example, the number density profile for Landau Damping test case at the finite time can be estimated with a lower variance compared to standard MC.

![Demo](examples/LandauDamping_init.png)

![Demo](examples/Damping_Ex2_Ey2.png)

---

## Features

- **Fully GPU-accelerated**: Uses CUDA to parallelize particle updates and field solvers.
- **Variance reduction (VR)**: Implements control variate methods to reduce noise in moment computations.
- **Importance weighting**: Dynamically adjusts particle weights using local Maxwellian-Boltzmann distribution as control variate.
- **Self-consistent field solving**: Solves the Poisson equation using a parallel Jacobi method.
- **Post-processing output**: Dumps moment fields for visualization and diagnostics.

---

## 🛠️ Build Instructions

### Requirements

- CUDA Toolkit (>= 11.x recommended)
- C++ compiler with C++11 or higher
- CMake

### Build

```bash
git clone https://github.com/yourusername/vlasov-poisson.git
cd vlasov-poisson
mkdir bin && cd bin
cmake ..
make
