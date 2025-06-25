# Sparse Regression with Quantum-Inspired and Classical Methods

This repository contains implementations and comparisons of sparse linear regression techniques, including:

- **LASSO** (Least Absolute Shrinkage and Selection Operator),
- **Orthogonal Matching Pursuit** (OMP),
- **Quantum Annealing** via QUBO formulation using D-Wave's Ocean SDK and QDeep Hybrid Solver.

The project measures accuracy, runtime, and memory consumption across these methods on synthetic datasets.

![Visualization](metrics.png)

## Problem Overview

We consider the standard sparse regression model:

$y = X\beta + \varepsilon$

where $X \in \mathbb{R}^{n \times p}$, $y \in \mathbb{R}^n$, and $\beta \in \mathbb{R}^p$ is a sparse vector of unknown coefficients. The goal is to recover $\beta$ from measurements $(X, y)$, assuming only a few non-zero entries in $\beta$.

## Methods

### LASSO

LASSO adds an $\ell_1$-penalty to enforce sparsity:

![LASSO formula](https://math.vercel.app/?from=%5Chat%7B%5Cbeta%7D_%7B%5Ctext%7BLASSO%7D%7D%20%3D%20%5Carg%5Cmin_%5Cbeta%20%5Cleft%5C%7B%20%5C%7Cy%20-%20X%5Cbeta%5C%7C_2%5E2%20%2B%20%5Clambda%20%5C%7C%5Cbeta%5C%7C_1%20%5Cright%5C%7D)

Implemented using `scikit-learn`'s `Lasso` with adjustable regularization parameter.

### Orthogonal Matching Pursuit (OMP)

OMP is a greedy algorithm that selects features one-by-one based on correlation with the residual. It performs well when features are orthogonal but degrades under strong multicollinearity.

### Quantum Annealing via QUBO

The regression problem is reformulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem. Each coefficient $\beta_j$ is discretized using signed binary encoding:

![Equation](https://math.vercel.app/?bgcolor=auto&from=%5Cbeta_j%20%3D%20%5Cdelta%20%5Ccdot%20%5Csum_%7Bk%3D0%7D%5E%7BK-1%7D%202%5Ek%20%28b%5E%2B_%7Bj%2Ck%7D%20-%20b%5E-_%7Bj%2Ck%7D%29%29%0A.svg)

where $b^+$ and $b^-$ are binary variables representing the positive and negative parts respectively, $\delta$ is the bit resolution, and $K$ is the number of bits per sign.

The resulting binary model is solved using:

- Simulated annealing (`neal.SimulatedAnnealingSampler`)
- Hybrid solvers from the QDeep SDK

In `Math_LASSO_to_QUBO.pdf`, you can find more detailed information on how to formulate the LASSO problem as a QUBO problem.
