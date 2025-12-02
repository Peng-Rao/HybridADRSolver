# Advection-Diffusion-Reaction Solver using deal.II

[![CI](https://github.com/Peng-Rao/HybridADRSolver/actions/workflows/ci.yml/badge.svg)](https://github.com/Peng-Rao/HybridADRSolver/actions/workflows/ci.yml)

This project implements a finite element solver for the steady-state advection-diffusion-reaction equation using
the [deal.II](https://www.dealii.org/) library.

## Problem Formulation

The solver addresses the following boundary value problem:

$$-\nabla \cdot (\mu \nabla u) + \nabla \cdot (\beta u) + \gamma u = f \quad \text{in } \Omega$$

$$u = g \quad \text{on } \Gamma_D \subset \partial\Omega \quad \text{(Dirichlet)}$$

$$\nabla u \cdot \mathbf{n} = h \quad \text{on } \Gamma_N = \partial\Omega \setminus \Gamma_D \quad \text{(Neumann)}$$

Where:

- $\mu$: Diffusion coefficient (scalar or tensor)
- $\beta$: Advection velocity field (vector)
- $\gamma$: Reaction coefficient (scalar, $\gamma \geq 0$)
- $f$: Source term
- $g$: Dirichlet boundary data
- $h$: Neumann boundary flux

## Weak Formulation

Multiplying by a test function $v \in H^1_0(\Omega)$ and integrating by parts:

$$\int_\Omega \mu \nabla u \cdot \nabla v \, dx + \int_\Omega (\beta \cdot \nabla u) v \, dx + \int_\Omega \gamma u v \, dx = \int_\Omega f v \, dx + \int_{\Gamma_N} h v \, ds$$

## Building

### Prerequisites

We recommend use [Spack](https://spack.io) to manage dependencies. Spack is a package manager for supercomputers, Linux,
macOS, and Windows. We can build deal.II automatically with Spack. Using the following commands to install Spack and
deal.II:

- Install Spack

```bash
git clone --depth=2 https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
```

- Install deal.II with MPI and other dependencies

```bash
spack install dealii +mpi +petsc +trilinos +p4est
```

### Compilation

- Load deal.II environment

```bash
spack load dealii
```

- Build the project

```bash
mkdir build
cd build
cmake ..
make -j
```

### Running

```bash
./GalerkinSolver    # Galerkin solver
```

## Mathematical Notes

### Coercivity and Stability

For well-posedness, we require:

1. $\mu > 0$ (positive diffusion)
2. $\gamma \geq 0$ (non-negative reaction)
3. $\nabla \cdot \beta$ bounded (divergence of convection)

The bilinear form is coercive if:

$$\gamma - \frac{1}{2}\nabla \cdot \beta \geq \gamma_0 > 0$$

### Error Estimates

For sufficiently smooth solutions, the standard Galerkin method gives:

$$\|u - u_h\|_{H^1} \leq C h^k |u|_{H^{k+1}}$$
