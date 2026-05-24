# PINOClosure — Physics-Informed Neural Operators for Chaotic PDE Long-Term Statistics

## Role in CozyPDE

This directory is a **read-only baseline reference** for the CozyPDE Agent. 

The original repository is:

- Repository: `neuraloperator/pino-closure-models`
- Paper: *Beyond Closure Models: Learning Chaotic Systems via Physics-Informed Neural Operators*
- Main idea: learn chaotic PDE dynamics with Physics-Informed Neural Operators, focusing not only on short-term trajectory accuracy but also on long-term statistical behavior.
- Relevant systems: 1D Kuramoto-Sivashinsky, 2D Kolmogorov Flow / Navier-Stokes, and baseline closure-style methods.

This repository must be treated as **method inspiration only**. Do not copy source files into `workspace/submission/code/`. Do not use any pretrained weights, generated external datasets, or solver-generated extra data from the original repository.

The PINO-closure idea is relevant because it directly addresses the following issues:

1. **Chaotic long-time rollout**
   - A model trained only with one-step or short-window MSE may look good early but drift into nonphysical long-time behavior.
   - PINO-style training adds physics residuals and statistics-oriented evaluation to regularize the learned dynamics.

2. **Known PDE but unknown test parameter**
   - The Agent can use the known KS equation during training while learning an internal parameter estimator for test-time rollout.
   
3. **Long-term statistics matter**
   - PINO-closure encourages the Agent to evaluate energy, spectrum, mean/std statistics, and distributional distances, not just MSE.
