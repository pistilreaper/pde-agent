# U-Net Baseline Reference

This directory contains reference materials for U-Net style PDE forecasting models.

U-Net is not a neural operator in the strictest sense, but it can be a strong convolutional baseline for fixed-resolution spatiotemporal prediction. In the competition materials, an official U-Net / U-Net-PF style checkpoint may be available for Task 1.

## Intended Use

The Agent may use this directory to understand:

- encoder-decoder convolutional architectures,
- skip connections,
- local spatial feature extraction,
- direct trajectory prediction,
- PDEBench-style training scripts,
- checkpoint fine-tuning for Task 1.

The Agent must not directly copy this directory into the final submission. Final code must be generated under `workspace/submission/code/`.

## Useful Files

Suggested layout:

- `source/unet1d.py`: 1D U-Net model definition.
- `source/train_burgers_unet.py`: reference training loop.
- `source/data_utils.py`: data loading and preprocessing.
- `configs/burgers_unet_config.yaml`: reference hyperparameters.
- `notes.md`: implementation notes and possible modifications.

## Key Ideas to Inspect

### 1. Local convolutional inductive bias

U-Net models are good at learning local spatial patterns. For Burgers dynamics, this may help capture shocks, steep gradients, and local transport.

Potential improvement directions:

- increase receptive field,
- use dilated convolution,
- add residual blocks,
- combine U-Net with temporal rollout,
- add spectral or derivative losses to compensate for local-only bias.

### 2. Skip connections

Skip connections preserve fine spatial details. The Agent should inspect whether skip connections help reconstruct high-gradient solution structures.

### 3. Direct versus rollout prediction

A U-Net may be used to predict:

- the next time step,
- a fixed future block,
- or the entire trajectory.

The Agent should evaluate which choice is more stable under the competition scoring rule.

## Task-specific Notes

For Task 1:
- Official U-Net checkpoint may be used if available.
- Fine-tuning can be useful if training time is limited.

For Task 2:
- Task 1 checkpoint must not be used.
- The model must be trained from scratch.
- Since test viscosity is unavailable, the Agent should avoid relying on explicit viscosity input at inference unless it also learns to infer it from the observed initial window.

## Recommended Agent Experiments

The Agent may consider:

1. Train a compact U-Net baseline quickly.
2. Compare against FNO on validation rollout.
3. Add residual prediction: predict increments instead of absolute states.
4. Add multi-step rollout loss.
5. Use U-Net as a refinement module after a coarse FNO prediction if time allows.

## Warnings

- Do not use external pretrained weights.
- Do not call numerical solvers.
- Do not generate extra data.
- Do not use Task 1 resources for Task 2.