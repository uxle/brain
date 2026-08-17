# `brain-regularization`

Production-grade regularization toolkit: dropout family, normalization layers, weight decay, explicit penalty regularizers, early stopping, and data augmentations for the Brain deep learning framework.

## Features

- **Dropout Family**:
  - `Dropout`: Inverted dropout ($p \in [0, 1)$), deterministic seed reproduction, in-place scaling, binary mask access, and fused `forward_add` residual integration.
  - `Dropout2d` / `Dropout3d`: Spatial & volumetric feature map dropout zeroing whole channels independently.
  - `AlphaDropout`: SELU self-normalizing network compatibility preserving zero mean and unit variance.
  - `ConcreteDropout`: Continuous relaxation of dropout with temperature annealing and learned parameter $p$ (Gal & Ghahramani).
  - `compute_mc_dropout_statistics`: Monte Carlo test-time sampling for epistemic uncertainty, predictive mean, variance, and confidence intervals.
- **Normalization Layers**:
  - `BatchNorm1d` / `BatchNorm2d` / `BatchNorm3d`: Running statistics accumulation via exponential moving averages ($\eta$), affine scale/shift ($\gamma, \beta$), numerical stability ($\epsilon$), and training/eval mode toggling.
  - `LayerNorm`: Feature dimension normalization with learnable affine parameters and fused residual addition.
  - `RMSNorm`: Root Mean Square Normalization for modern high-performance Large Language Models (LLMs).
  - `GroupNorm`: Channels partitioned into $G$ independent groups for batch-size invariant normalization.
  - `InstanceNorm1d` / `InstanceNorm2d` / `InstanceNorm3d`: Per-sample channel normalization for style transfer and generative models.
  - `WeightNorm`: Magnitude-direction reparameterization $w = g \cdot v / \|v\|$.
  - `SpectralNorm`: Power iteration spectral radius estimation $\sigma(W)$ bounding Lipschitz continuity.
- **Explicit Regularizers & Weight Decay**:
  - `L1Regularizer` (Lasso): Enforces parameter sparsity with subgradient penalties.
  - `L2Regularizer` (Ridge): Penalizes large parameter norms with isotropic shrinkage.
  - `ElasticNetRegularizer`: Combined convex $L_1 + L_2$ penalty formulation.
  - `DecoupledWeightDecay`: AdamW / SGDW style direct parameter decay $w \leftarrow w \cdot (1 - \eta \lambda)$.
- **Termination & Training Policies**:
  - `EarlyStopping`: Monitored validation metric tracker with patience counters, minimum delta thresholds, Min/Max modes, and best-checkpoint restoration.
  - `StopOnPlateau`, `StopOnBudget`, `CompositeStopPolicy`: Flexible training termination rules.
  - `CurriculumScheduler`: Progressive regularization annealing (e.g. ramping dropout probability $p(t)$).
  - `Mixup`, `Cutout`, `CutMix`: Tensor-level implicit data augmentations.
  - `GaussianNoise`, `apply_fgsm_perturbation`: Jitter injection and Fast Gradient Sign Method adversarial regularization.
  - `LabelSmoothing`: Soft target distribution generation for classification targets.
  - `compute_consistency_loss`: Pi-model stochastic consistency penalty.
  - `RegStack`: Multi-regularizer composition with independent loss weights.
  - `RegHook`: Training loop interceptor integrating with `brain-optim`.
- **Zero Runtime Dependencies**: Pure standard library (`std`) and `brain-core` only. Edition 2021 stable Rust.

## Quick Start

```rust
use brain_core::Tensor;
use brain_regularization::prelude::*;

// 1. Setup Inverted Dropout
let mut dropout = Dropout::new(0.2);
let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
let dropped = dropout.apply(&x).unwrap();

// Switch to evaluation mode
dropout.eval_mode();
let eval_out = dropout.apply(&x).unwrap();
assert_eq!(eval_out.data(), x.data());

// 2. Setup Layer Normalization
let ln_config = LayerNormConfig {
    normalized_shape: vec![4],
    eps: 1e-5,
    elementwise_affine: true,
};
let ln = LayerNorm::new(ln_config);
let normalized = ln.forward(&x).unwrap();

// 3. Setup Early Stopping
let mut early_stop = EarlyStopping::new(EarlyStopConfig {
    patience: 3,
    min_delta: 1e-3,
    mode: MetricMode::Min,
    restore_best_weights: true,
});

let should_halt = early_stop.step(0, 0.45, None);
assert!(!should_halt);
```
