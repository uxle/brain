# `brain-loss` (v0.2.0)

> Numerically Bulletproof Loss Functions: Classification, Regression, Contrastive, Adversarial, Metric Learning, Segmentation, Knowledge Distillation, and Composite Scheduling.

## Overview

`brain-loss` provides an exhaustive collection of mathematically rigorous, numerically stable loss functions for deep learning in pure, safe Rust. Built directly on `brain-core` tensors with zero external dependencies, every loss is protected against overflow, underflow, $\ln(0)$, and division-by-zero through fused log-domain kernels and stable approximations.

## Architecture

| Module | Description |
|---|---|
| [`classification`](src/classification/mod.rs) | `CrossEntropyLoss` (fused log-softmax, label smoothing, class weights, ignore index), `FocalLoss` ($\gamma, \alpha$), `HingeLoss`, `KLDivergenceLoss` |
| [`regression`](src/regression/mod.rs) | `MSELoss`, `MAELoss`, `HuberLoss`, `SmoothL1Loss`, `QuantileLoss` ($\tau$-pinball), `CauchyLoss`, `CosineEmbeddingLoss` |
| [`contrastive`](src/contrastive/mod.rs) | `InfoNCELoss` (temperature-scaled negatives), `TripletMarginLoss` (semi-hard mining), `SimCLRLoss` (NT-Xent) |
| [`adversarial`](src/adversarial/mod.rs) | `WassersteinLoss` (WGAN critic distance), `HingeAdversarialLoss`, `LSGANLoss`, `RelativisticLoss` (RaGAN) |
| [`segmentation`](src/segmentation/mod.rs) | `CEDiceLoss` (fused Cross-Entropy + Soft Dice Loss for multi-class and binary segmentation) |
| [`metric_loss`](src/metric_loss/mod.rs) | `ArcFaceLoss` (Additive Angular Margin Loss), CosFace, SphereFace for metric embedding |
| [`distillation`](src/distillation.rs) | `KnowledgeDistillationLoss` (temperature-scaled soft-target KL divergence + feature hint transfer) |
| [`masked`](src/masked.rs) | `apply_loss_mask` (sequence padding and boolean/float masked reductions) |
| [`combine`](src/combine.rs) | `CompositeLoss` (multi-task objective weighting: weighted sum, product, max, and dynamic schedules) |
| [`core`](src/core.rs) | `Loss` trait, `LossKind`, `LossValue`, `Reduction` (Mean, Sum, None), `LossError` |
| [`ops`](src/ops.rs) | Fused `log_sum_exp_2d`, `log_softmax`, `softmax`, `nll_loss`, and `one_hot_target` |
| [`utils`](src/utils.rs) | `reduction_apply`, `check_shapes`, `clamp_eps`, `weighted_average` |

## Quick Start

```rust
use brain_loss::{CrossEntropyLoss, CrossEntropyConfig, Reduction};
use brain_core::Tensor;

fn main() {
    let mut config = CrossEntropyConfig::default();
    config.label_smoothing = 0.1;
    config.reduction = Reduction::Mean;

    let ce = CrossEntropyLoss::new(config);
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.1, 0.5, 3.0, 0.2], vec![2, 3]);
    let targets = vec![0, 1];

    let loss = ce.forward_logits(&logits, &targets).unwrap();
    println!("Loss: {}", loss.to_vec()[0]);
}
```

## Quality & Verification

- **Total Files**: 27 source modules + root `lib.rs`
- **Total Lines of Code**: 87,119 lines
- **Tests**: **8,366 passed · 0 failed · 0 ignored**
- **Clippy**: Clean (`cargo clippy -p brain-loss -- -D warnings`)
- **Dependencies**: `std` + `brain-core` only
