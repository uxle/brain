# `brain-gan`

Generative adversarial network framework — DCGAN/ResNet/conditional/PatchGAN, losses, training, StyleGAN-lite, and CycleGAN-lite — in 100% safe, zero-dependency Rust.

## Overview

`brain-gan` implements GAN architectures and training stabilizers for the Brain ecosystem on `brain-core` tensors: DCGAN, ResNet, and class-conditional generators; DCGAN, conditional, and Patch discriminators; five classic loss families plus perceptual losses; WGAN-GP and R1 gradient penalties; spectral normalization and weight clipping; and EMA utilities. The unified `Gan` model performs `train_step`/`sample`/`evaluate` in a few calls, while `StyleGanLite` (mapping network + AdaIN + style mixing) and `CycleGanLite` (cycle-consistency/identity losses) provide advanced architectures.

## Features

- **Generators**: `Generator` trait, `DcganGenerator`, `ResnetGenerator`, `ConditionalGenerator`, `sample_latent`.
- **Discriminators**: `DcganDiscriminator`, `ConditionalDiscriminator`, `PatchDiscriminator`.
- **Losses**: `bce_loss_d`, `lsgan_loss_d`, `hinge_loss_d`/`hinge_loss_g`, `wgan_loss_d`/`wgan_loss_g` (via `LossVariant`), plus perceptual `perceptual_loss`, `feature_matching_loss`, `gram_matrix`.
- **Training**: `GanTrainer` + `TrainLoop`, `gradient_penalty`/`r1_penalty` (`PenaltyConfig`), `GanTrainStats`.
- **Unified model**: `Gan::new(config)` with `train_step(&real_batch) -> GanMetrics`, `sample(n, seed)`, `evaluate(&real) -> GanEvalReport`, `summary()`.
- **Advanced**: `StyleGanLite` with `MappingNetwork`, `adaptive_instance_norm`, `style_mix`; `CycleGanLite` with `cycle_consistency_loss`/`cycle_total_loss`/`identity_loss`.
- **Evaluation & utils**: `fid_lite`, `is_lite`, `eval_gan`, `assemble_grid`, `interpolate_latents`, EMA (`track_ema`), seeding, Gaussian sampling (`box_muller`, `sample_gaussian`).

## Modules

| Module | Contents |
|---|---|
| `generator/` | `dcgan`, `resnet`, `conditional` |
| `discriminator/` | `dcgan`, `conditional`, `patch` |
| `losses/` | `classic`, `perceptual` |
| `train/` | `mod` (`GanTrainer`), `loop_` (`TrainLoop`), `penalties` |
| `eval/` | `mod` (`fid_lite`, `is_lite`, `eval_gan`), `samples` (grids, interpolation) |
| `stylegan_lite.rs` | `MappingNetwork`, `StyleGanLite`, AdaIN, style mixing |
| `cycle.rs` | `CycleGanLite`, cycle-consistency/identity losses |
| `gan/` | Unified `Gan` model |
| `core.rs`, `config.rs`, `ops.rs`, `utils.rs`, `builder.rs` | `GanState`/`GanMetrics`, `GanConfig`, activations/spectral norm, seeds/EMA, `GanBuilder` |

## Quick Start

```rust
use brain_core::Tensor;
use brain_gan::{Gan, GanConfig};

let mut gan = Gan::new(GanConfig::default());
let metrics = gan.train_step(&Tensor::zeros(vec![1, 3, 32, 32]));
println!("D loss: {:.4}, G loss: {:.4}", metrics.d_loss, metrics.g_loss);

let samples = gan.sample(4, 42); // Vec<Tensor> of generated images
```

## Testing

```bash
cargo test -p brain-gan -j 2
```

## Workspace Role

Generative adversarial network family for the Brain ecosystem. Depends only on `brain-core` (tensors) — zero external dependencies, 100% safe Rust.