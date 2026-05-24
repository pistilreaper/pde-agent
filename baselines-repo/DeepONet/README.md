# DeepONet Baseline Reference

This directory contains reference materials for Deep Operator Network style models.

DeepONet represents an operator using two components:

- a branch network that encodes the input function or initial condition,
- a trunk network that encodes query coordinates,
- an inner-product style combination that produces the solution value.

## Intended Use

The Agent may use this directory to understand:

- branch/trunk decomposition,
- coordinate-conditioned prediction,
- operator learning formulation,
- how to represent spatial and temporal coordinates,
- how DeepONet differs from FNO and convolutional models.

The Agent must not directly copy this directory into the final submission. Final code must be generated under `workspace/submission/code/`.

## Useful Files

Suggested layout:

- `source/deeponet_model.py`: branch/trunk architecture.
- `source/train_deeponet.py`: reference training loop.
- `source/data_utils.py`: coordinate sampling and batching.
- `configs/burgers_deeponet_config.yaml`: reference hyperparameters.
- `notes.md`: summary of transferable ideas.

## Key Ideas to Inspect

### 1. Branch network

The branch network encodes the observed initial condition or short time window. For this competition, the Agent should consider whether the branch input should be:

- the first state only,
- the first 10 states,
- a compressed temporal representation,
- or features derived from the initial window.

### 2. Trunk network

The trunk network encodes coordinates such as:

- spatial coordinate `x`,
- time coordinate `t`,
- optionally normalized time-step index.

Coordinate conditioning may help direct full-trajectory prediction, but it can be slower than grid-based models if implemented inefficiently.

### 3. Output formulation

Possible output modes:

- predict each `(t, x)` point by coordinate query,
- predict full spatial fields for selected times,
- use DeepONet as a correction or latent parameter estimator.

For this competition, full coordinate-wise DeepONet may be too slow for inference unless carefully vectorized.

## Task-specific Notes

For Task 1:
- DeepONet can be used as an alternative to FNO/U-Net, but training time should be considered.
- It may be useful for coordinate-conditioned refinement.

For Task 2:
- The model must be trained from scratch.
- If viscosity values are available during training, the Agent may use them as auxiliary labels or latent supervision, but inference must not require test viscosity.

## Recommended Agent Experiments

The Agent may consider:

1. Use DeepONet concepts to design a coordinate-conditioned decoder.
2. Use the initial window as the branch input.
3. Train a latent parameter estimator from the initial window.
4. Compare direct trajectory prediction against grid-based FNO/U-Net.
5. Avoid slow pointwise inference; prefer vectorized batch evaluation.

## Warnings

- Do not use external pretrained weights.
- Do not query numerical solvers.
- Do not generate additional trajectories.
- Do not let inference exceed 2 minutes.