# Physics-informed DeepONet Baseline Reference

This directory contains reference materials for physics-informed DeepONet style models.

Physics-informed methods add PDE residual constraints to data fitting. For Burgers equation, possible residual terms involve temporal derivatives, spatial derivatives, nonlinear advection, and viscosity-dependent diffusion.

## Intended Use

The Agent may use this directory to understand:

- how physics residuals are formed,
- how automatic differentiation or finite differences are used,
- how data loss and physics loss are balanced,
- how physics constraints may improve long-horizon stability.

The Agent must not directly copy this directory into the final submission. Final code must be generated under `workspace/submission/code/`.

## Useful Files

Suggested layout:

- `source/pi_deeponet_model.py`: physics-informed DeepONet model.
- `source/physics_loss.py`: residual loss construction.
- `source/train_pi_deeponet.py`: reference training loop.
- `configs/pi_deeponet_burgers_config.yaml`: reference hyperparameters.
- `notes.md`: curated implementation notes.

## Key Ideas to Inspect

### 1. Burgers residual

A typical 1D viscous Burgers equation has the form:

`u_t + u * u_x = nu * u_xx`

Depending on the dataset convention, signs and coefficients must be verified from the official data description before use.

Physics-informed loss may penalize:

- temporal derivative mismatch,
- nonlinear advection residual,
- diffusion residual,
- spatial smoothness inconsistency.

### 2. Derivative computation

The Agent may consider:

- finite differences on the predicted grid,
- spectral derivatives,
- automatic differentiation if the model is coordinate-based.

For grid-based FNO/U-Net models, finite-difference or spectral derivative losses are usually easier than automatic differentiation.

### 3. Loss balancing

Physics loss can help long-horizon stability, but excessive physics weighting may hurt short-term accuracy. The Agent should tune the balance carefully.

Possible losses:

- data MSE,
- relative MSE,
- multi-step rollout MSE,
- derivative loss,
- Burgers residual loss,
- spectral loss.

## Task-specific Notes

For Task 1:
- If viscosity is fixed and known, physics residual loss may be straightforward.
- Official checkpoint fine-tuning may be combined with a light physics regularization stage.

For Task 2:
- Test viscosity is unavailable.
- If viscosity values are provided in training data, the Agent may train an auxiliary viscosity estimator from the observed initial window.
- Physics residuals requiring explicit viscosity at inference should be avoided unless viscosity is internally inferred by the model.

## Recommended Agent Experiments

The Agent may consider:

1. Add a lightweight finite-difference residual loss.
2. Use spectral derivative loss for smoothness and phase stability.
3. Add multi-step rollout loss before adding heavy physics loss.
4. For Task 2, train a latent viscosity estimator only from the initial window.
5. Check whether physics loss improves validation rollout, not only training loss.

## Warnings

- Physics residuals must not call a numerical solver.
- Do not generate extra trajectories.
- Do not assume viscosity is available at Task 2 inference.
- Do not overfit the score by manually editing predictions outside the Agent workflow.