# FNO Baseline Reference

This directory contains reference materials for Fourier Neural Operator style models.

FNO is one of the most relevant baselines for the Burgers tasks because it learns mappings between discretized function spaces and can exploit spectral structure in periodic or smooth PDE fields.

## Intended Use

The Agent may use this directory to understand:

- spectral convolution,
- Fourier mode truncation,
- lifting and projection layers,
- autoregressive rollout,
- multi-step PDE trajectory prediction,
- resolution handling,
- training and inference patterns for Burgers data.

The Agent must not directly copy this directory into the final submission. Final code must be generated under `workspace/submission/code/`.

## Useful Files

Suggested layout:

- `source/fno_model.py`: core FNO model definition.
- `source/spectral_conv.py`: spectral convolution implementation.
- `source/train_burgers.py`: reference training loop.
- `source/data_utils.py`: data loading and normalization patterns.
- `configs/burgers_fno_config.yaml`: reference hyperparameters.
- `notes.md`: human-curated notes about important implementation details.

## Key Ideas to Inspect

### 1. Spectral convolution

FNO applies convolution in Fourier space. For 1D Burgers, the Agent should inspect how the model:

- applies FFT along the spatial dimension,
- keeps only low-frequency modes,
- learns complex-valued spectral weights,
- transforms the result back to physical space.

Potential improvement directions:

- tune number of Fourier modes,
- tune channel width,
- use residual connections,
- improve normalization,
- add spectral-domain loss terms,
- stabilize long rollout with multi-step loss.

### 2. Autoregressive rollout

The competition requires predicting future time steps from an initial window. The Agent should inspect whether the baseline predicts:

- one step at a time,
- multiple steps at once,
- or the whole trajectory directly.

For long-horizon performance, one-step training may be insufficient because rollout errors accumulate. The Agent should consider multi-step training or scheduled sampling if time permits.

### 3. Task-specific constraints

For Task 1:
- Official PDEBench FNO checkpoint may be used for fine-tuning.
- The Agent may compare fine-tuning against training from scratch.

For Task 2:
- No Task 1 checkpoint is allowed.
- The model must be trained from scratch using only Task 2 data.
- Since test viscosity is unavailable, the Agent should infer physical behavior from the initial condition alone or train a latent-parameter model.

## Recommended Agent Experiments

The Agent may consider:

1. Reproduce a minimal FNO training pipeline.
2. Add multi-step rollout loss.
3. Compare direct trajectory prediction against autoregressive prediction.
4. Add derivative or spectral smoothness regularization.
5. Use validation rollout error, not only one-step error, for model selection.
6. Keep inference under 2 minutes.

## Warnings

- Do not call numerical solvers.
- Do not generate synthetic PDE trajectories.
- Do not use Task 1 checkpoint for Task 2.
- Do not rely on files outside official data, checkpoint, baseline reference, and Agent-generated code.