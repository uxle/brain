# `brain-diffusion` (v0.2.0)

> Production-Grade Diffusion Models: DDPM, DDIM, DPMSolver, CFG, Noise Schedules, and U-Net Backbones.

## Overview

`brain-diffusion` provides a complete suite of generative diffusion algorithms. It implements discrete and continuous noise schedules, deterministic and stochastic ODE/SDE samplers (DDPM, DDIM, DPMSolver), classifier-free guidance, and multi-scale U-Net backbones with cross-attention.

## Architecture

| Module | Description |
|---|---|
| `schedule` | Linear, cosine, sigmoid, and exponential variance schedules ($eta_t, lpha_t, ar{lpha}_t$) |
| `sampler` | DDPM, DDIM (accelerated steps), DPMSolver (fast higher-order ODE solver) |
| `unet` | Denoising U-Net: ResNet blocks, self-attention, cross-attention, time-step embeddings |
| `guidance` | Classifier-free guidance (CFG) mixing and conditioning projections |
| `pipeline` | End-to-end image generation pipelines with progressive denoising loop |

## Quality & Verification

- **Tests**: 11,851 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-diffusion -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
