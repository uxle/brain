# `brain-optim`

Production-grade optimizer suite, learning rate schedulers, gradient clipping engine, dynamic mixed-precision loss scaling, stochastic weight averaging (SWA), and parameter EMA for the Brain deep learning framework.

## Features

- **Optimizers (12+ Implementations)**:
  - `Sgd` & `SgdNesterov`: Stochastic Gradient Descent with classical/Nesterov momentum, dampening, L2 weight decay, and decoupled weight decay (SGDW).
  - `Adam` & `AdamW`: Adaptive Moment Estimation with first/second moment bias correction, AMSGrad variant, and decoupled weight decay.
  - `Adamax`: Infinity-norm variant of Adam for models with embeddings and sparse updates.
  - `Nadam`: Nesterov-accelerated Adaptive Moment Estimation.
  - `RAdam`: Variance-rectified Adam stabilizing early training dynamics without warmup.
  - `Lamb`: Layer-wise Adaptive Moments for large batch distributed training with trust ratio scaling.
  - `Lion`: EvoLved Sign Momentum optimizer with sign-based updates and memory efficiency.
  - `NovoGrad`: Normalized stochastic gradient descent with layer-wise second moment reduction.
  - `Rmsprop`: Root Mean Square Propagation with centered gradient option and momentum.
  - `Adagrad`: Adaptive Gradient Algorithm with per-parameter historical gradient accumulators.
  - `Adadelta`: Learning rate-free adaptive optimizer dynamically scaled by running averages.
- **Learning Rate Schedulers (8+ Variants)**:
  - `StepLR`, `MultiStepLR`, `ExponentialLR`, `PolynomialLR`.
  - `CosineAnnealingLR` & `CosineAnnealingWarmRestarts` (SGDR).
  - `CyclicLR` (Triangular, Triangular2, ExpRange).
  - `OneCycleLR` (1cycle policy with 2-phase/3-phase cosine/linear annealing).
  - `LinearWarmup`, `ConstantWarmup`, `ExponentialWarmup`.
  - `ReduceLROnPlateau` (Min/Max evaluation modes, relative/absolute thresholds, cooldown).
  - `ChainedScheduler` for composite multi-stage learning rate schedules.
- **Gradient Clipping**:
  - `clip_grad_norm_`: L1, L2, and $L_\infty$ global gradient norm clipping.
  - `clip_grad_value_`: Value clipping in range $[-c, c]$.
  - `AGC` / `clip_grad_adaptive_`: Adaptive Gradient Clipping based on weight-to-gradient norm ratios.
- **Training Dynamics & Regularization**:
  - `GradScaler`: Automatic mixed precision dynamic loss scaler with growth and backoff.
  - `ModelEma`: Exponential moving average of model parameters with warmup.
  - `SwAOptimizer`: Stochastic Weight Averaging for ensembling flat loss minima.
  - `Lookahead`: Fast/slow weight interpolation wrapper.
  - `Sam` / `Asam`: Sharpness-Aware Minimization for flat loss landscape convergence.
  - `LrFinder`: Range test with smoothed loss trajectory and minimum gradient recommendation.
  - `loss_landscape`: 1D/2D parameter interpolation and filter-normalized direction generation.
- **Architecture**:
  - Zero external runtime dependencies (pure Rust std).
  - 100% test coverage with 8,739 tests.
  - Full parameter group customization (`ParamGroup`, learning rate multipliers, custom options).

## Quick Start

```rust
use brain_core::Tensor;
use brain_optim::prelude::*;
use brain_optim::ParamGroup;

// 1. Setup parameter groups
let param_group = ParamGroup::new(vec![0], 1e-3)
    .with_weight_decay(1e-2)
    .with_betas(0.9, 0.999);

// 2. Instantiate optimizer via builder
let mut optimizer = OptimizerBuilder::new()
    .adamw()
    .lr(1e-3)
    .weight_decay(1e-2)
    .add_param_group(param_group);

// 3. Perform optimization step
let mut params = vec![Tensor::from_slice(&[1.0, -2.0], vec![2])];
let grads = vec![Tensor::from_slice(&[0.1, -0.2], vec![2])];

let mut opt = Adam::adamw(vec![ParamGroup::new(vec![0], 1e-3)], 1e-3, 1e-2);
let step_info = opt.step(&mut params, &grads).unwrap();
println!("Completed step {}, updated {} params", step_info.step_count, step_info.num_params_updated);
```
