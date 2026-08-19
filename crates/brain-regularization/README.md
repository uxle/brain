# `brain-regularization`

> Dropout family, normalization layers, weight regularizers, early stopping, label smoothing, curriculum, and augmentation — in safe Rust.

## Overview

`brain-regularization` supplies the regularization toolbox for training: inverted `Dropout` / `Dropout2d` with eval mode, alpha and concrete (adaptive) dropout, spectral and weight normalization, L1/L2/ElasticNet regularizers, decoupled weight decay, mixup/cutout augmentation, label smoothing, early stopping and plateau/budget stop rules, and a curriculum scheduler — all operating on `brain-core` tensors.

## Features

- `Dropout` (seeded, inverted scaling, `eval_mode`, residual `forward_add`) and `Dropout2d`; `AlphaDropout`, `ConcreteDropout` with `current_p`
- Normalization: BatchNorm, LayerNorm (+ residual), RMSNorm, GroupNorm, InstanceNorm (`normalization` module), plus `SpectralNorm` and `WeightNorm` weight reparameterizations
- Regularizers: `L1Regularizer`, `L2Regularizer`, `ElasticNetRegularizer` over parameter tensors; `DecoupledWeightDecay::apply_decay`
- Training controls: `EarlyStopping::step` with configurable patience/restore, `StopOnPlateau`, `StopOnBudget`, `CurriculumScheduler::get_value`, `RegHook` train hooks
- Augmentation & robustness: `Mixup`, `Cutout`, noise injection, FGSM perturbation, MC-dropout uncertainty (`compute_mc_dropout_statistics`), consistency loss
- `LabelSmoothing::smooth_targets`, `RegularizerSet` penalty aggregation, `RegRegistry::parse_kind` string-based kind resolution, `RegConfig`/`DropoutConfig`/`NormConfig`

## Modules

| Module | Description |
|---|---|
| `dropout` | `Dropout`, `Dropout2d`, `AlphaDropout`, `ConcreteDropout` |
| `normalization` | BatchNorm, LayerNorm, RMSNorm, GroupNorm, InstanceNorm, SpectralNorm, WeightNorm |
| `regularizers` | L1 / L2 / ElasticNet regularizers |
| `decay` | `DecoupledWeightDecay` |
| `earlystop` | `EarlyStopping`, `EarlyStopConfig`, `EarlyStopState` |
| `stopping` | `StopOnBudget`, `StopOnPlateau` |
| `label_smooth` | `LabelSmoothing` |
| `curriculum` | `CurriculumScheduler` |
| `augment` | `Mixup`, `Cutout`, implicit-regularization config |
| `perturb` | Noise injection, FGSM perturbation |
| `dropout_uncertainty` | MC-dropout statistics |
| `consistency` | Consistency loss |
| `rules` | `RegularizerSet` weighted penalties |
| `registry` | `RegRegistry::parse_kind` |
| `train_hooks` | `RegHook` post-optimizer-step hook |
| `core` / `config` / `utils` / `ops` | `RegState`, configs, RNG, tensor-level helpers |

## Quick Start

```rust
use brain_regularization::dropout::Dropout;

let mut dropout = Dropout::with_seed(0.5, 42);
let x = brain_core::Tensor::from_slice(&[1.0; 16], vec![4, 4]);
let train_out = dropout.apply(&x).expect("dropout train");
dropout.eval_mode();
let eval_out = dropout.apply(&x).expect("dropout eval"); // identity
```

## Testing

```bash
cargo test -p brain-regularization -j 2
```

## Workspace Role

Depends only on `brain-core`. `brain-regularization` slots between `brain-nn` and `brain-train`: layers are wrapped in dropout/norm/weight regularizers, and training hooks keep early stopping, decay, and curriculum in lockstep with optimizer steps.