# `brain-loss`

Pure-Rust loss function library: classification, regression, contrastive, adversarial, metric, segmentation, and distillation losses.

## Overview

`brain-loss` implements numerically robust loss functions over `brain-core` tensors, with a `Loss` trait that exposes both plain-tensor `forward` and differentiable `forward_value` (over `brain-autograd::Value`) so losses can drive autograd training. It includes a composite-loss orchestrator, masked losses, and fused softmax/NLL helpers.

## Features

- **Classification** — `CrossEntropyLoss` (with label smoothing), `FocalLoss`, `HingeLoss`, `KLDivergenceLoss`.
- **Regression** — `MSELoss`, `MAELoss`, `HuberLoss`, `SmoothL1Loss`, `QuantileLoss`, `CauchyLoss`, `CosineEmbeddingLoss`, `AngularDistanceLoss`.
- **Contrastive & metric** — `InfoNCELoss`, `SimCLRLoss` (NT-Xent), `TripletMarginLoss`, `ContrastiveLoss`, `ArcFaceLoss`.
- **Adversarial** — `WassersteinLoss`, `HingeAdversarialLoss`, `LSGANLoss`, `RelativisticLoss`.
- **Segmentation & distillation** — `CEDiceLoss` (cross-entropy + soft Dice), `KnowledgeDistillationLoss` (temperature-scaled soft targets).
- **Composition** — `CompositeLoss` orchestration (weighted sum, product, max), `apply_loss_mask` padding-aware reduction.
- **Fused helpers** — `softmax`, `log_softmax`, `nll_loss`, `one_hot_target`, `log_sum_exp_2d`.
- **Differentiable** — `Loss::forward_value(pred: &Value, target: &Tensor) -> LossResult<Value>` builds a differentiable loss node.

## Modules

| Module | Description |
|---|---|
| `core` | `Loss` trait, `LossKind`, `LossValue`, `Reduction`, `LossError` |
| `classification` | Cross-entropy, focal, hinge, KL divergence |
| `regression` | MSE, MAE, Huber, SmoothL1, quantile, Cauchy, cosine-embedding |
| `contrastive` | InfoNCE, SimCLR, triplet margin, contrastive |
| `adversarial` | WGAN, hinge, LSGAN, relativistic |
| `metric_loss` | ArcFace angular margin |
| `segmentation` | Combined cross-entropy + soft Dice |
| `distillation` | Temperature-scaled knowledge distillation |
| `combine` | Weighted composite loss orchestrator |
| `masked` | Masked loss wrappers |
| `ops` | Fused softmax, log-softmax, NLL, one-hot |
| `utils` | Shape verification, reduction, clamping helpers |

## Quick Start

```rust
use brain_core::Tensor;
use brain_loss::classification::{CrossEntropyConfig, CrossEntropyLoss};
use brain_loss::core::Reduction;

let ce = CrossEntropyLoss::new(CrossEntropyConfig {
    reduction: Reduction::Mean,
    label_smoothing: 0.0,
    ..Default::default()
});

let logits = Tensor::from_slice(&[2.0, 1.0, 0.1, 0.5, 3.0, 0.2], vec![2, 3]);
let targets = vec![0, 1];
let loss = ce.forward_logits(&logits, &targets).unwrap();
println!("loss = {}", loss.get(0));
```

## Testing

```bash
cargo test -p brain-loss --test loss_test -j 2
cargo test -p brain-loss -j 2
```

## Workspace Role

Depends on `brain-core` and `brain-autograd`. Consumers: `brain-train`, `brain-cli`, and the `brain` facade (via its `loss` feature).
