# `brain-optim`

Pure-Rust optimizer suite: SGD/Adam families, LR schedulers, gradient clipping, and training accelerators.

## Overview

`brain-optim` implements 10+ numerical optimizers over plain `brain-core` tensors through a shared `Optimizer` trait, plus learning-rate schedulers, gradient clipping/norm adaptive rules, and training accelerators (AMP grad scaling, SWA, Lookahead, SAM, EMA). Optimizer state can be checkpointed via `StateDict`.

## Features

- **Optimizers** — `Sgd` (+ Nesterov via `SgdNesterov`), `Adam`, `AdamW`, `Adamax`, `Nadam`, `RAdam`, `Lamb`, `Lion`, `NovoGrad`, `Rmsprop`, `Adagrad`, `Adadelta`.
- **LR schedulers** — `StepLR`, `MultiStepLR`, `ExponentialLR`, `PolynomialLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `CyclicLR`, `OneCycleLR`, `ReduceLROnPlateau`, and warmup schedulers (`LinearWarmup`, `ConstantWarmup`, `ExponentialWarmup`) via the `LrScheduler` trait.
- **Gradient clipping** — `clip_grad_norm_`, `clip_grad_value_`, `clip_grad_adaptive_`, adaptive gradient clipping (`AGC`), `GradClipper`, `NormType`.
- **Accelerators** — `GradScaler` (mixed-precision loss scaling), `SwAOptimizer` (stochastic weight averaging), `Lookahead`, `Sam` (sharpness-aware minimization), `ModelEma`.
- **Analysis & tooling** — `LrFinder` (LR range test), loss-landscape interpolation (`interpolate_1d`, `create_filter_normalized_direction`), `OptimizerBuilder` / `OptimizerKind` factory.
- **State management** — `StateDict` and `OptimizerCheckpoint` serialization; `ParamGroup` / `ParamId` parameter grouping.

## Modules

| Module | Description |
|---|---|
| `optimizer` | `Optimizer` trait, `ParamGroup`, `StepInfo`, `OptimResult` |
| `sgd` / `adam` / `radam` / `lamb` / `lion` / `novograd` / `rmsprop` / `adagrad` / `adadelta` | Optimizer implementations |
| `schedulers` | LR scheduler trait and 12 scheduler implementations |
| `clipping` | Norm/value/adaptive clipping and AGC |
| `amp` / `ema` / `lookahead` / `sam` / `swa` | Training accelerators |
| `lr_finder` | Learning-rate range finder |
| `state` | `StateDict`, `OptimizerCheckpoint` |
| `builder` | `OptimizerBuilder` factory |
| `loss_landscape` | Loss landscape analysis helpers |

## Quick Start

```rust
use brain_core::Tensor;
use brain_optim::{Optimizer, ParamGroup, Sgd, SgdConfig};

let mut params = vec![Tensor::from_slice(&[1.0], vec![1])];
let grads = vec![Tensor::from_slice(&[2.0], vec![1])];

let group = ParamGroup::new(vec![0], 0.1);
let mut opt = Sgd::new(vec![group], SgdConfig { lr: 0.1, ..Default::default() });
opt.step(&mut params, &grads).unwrap();
println!("theta = {}", params[0].get(0)); // 0.8
```

## Testing

```bash
cargo test -p brain-optim --test optim_step_test -j 2
cargo test -p brain-optim --test autopilot_regression -j 2
cargo test -p brain-optim -j 2
```

`optim_step_test` checks exact closed-form multi-step trajectories against hand-computed references.

## Workspace Role

Depends on `brain-core`. Consumers: `brain-train`, `brain-cli`, `brain-rl`, and the `brain` facade (via its `optim` feature).
