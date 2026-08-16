# `brain-gan` (v0.2.0)

> Production-Grade GAN Framework: DCGAN, ResNet, Conditional, PatchGAN, StyleGAN-Lite, CycleGAN-Lite, WGAN-GP, and Evaluation.

## Overview

`brain-gan` delivers generative adversarial network architectures and training stabilizers. It includes DCGAN, ResNet-style, class-conditional, and PatchGAN generators & discriminators, 5 loss variants (BCE, LSGAN, Hinge, WGAN, Relativistic), WGAN-GP and R1 gradient penalties, spectral normalization, FID-lite and IS-lite evaluation metrics, StyleGAN-lite (mapping network + AdaIN), and CycleGAN-lite cycle-consistency formulation.

## Architecture

| Module | Description |
|---|---|
| `generator` | `Generator` trait, DCGAN transposed-conv stack, ResNet generator, Conditional cGAN generator |
| `discriminator` | `Discriminator` trait, DCGAN stride-conv stack, cGAN projection discriminator, PatchGAN (70x70) |
| `losses` | BCE minimax, Least Squares (LSGAN), Hinge, Wasserstein (WGAN), Relativistic (RaGAN), Perceptual |
| `train` | `GanTrainer`, alternating D/G steps ($n_\text{critic}$), WGAN-GP finite-difference penalty, R1/R2 penalties |
| `eval` | Inception-free IS-lite, FID-lite feature covariance distance, fixed latent sampling, grid assembly |
| `stylegan_lite` | Disentangled $z \to w$ mapping network, Adaptive Instance Normalization (AdaIN), style mixing |
| `cycle` | CycleGAN paired/unpaired generator coordination, cycle consistency loss, identity loss |

## Quality & Verification

- **Tests**: 5,603 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-gan -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
