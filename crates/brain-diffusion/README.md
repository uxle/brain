# `brain-diffusion`

Deep generative diffusion suite — noise schedules, samplers, guidance, U-Net backbones, and training — in 100% safe, zero-dependency Rust.

## Overview

`brain-diffusion` implements the diffusion-model stack for the Brain ecosystem on top of `brain-core` tensors: linear/cosine/scaled-linear noise schedules behind a `NoiseSchedule` trait, four multistep samplers (DDPM, DDIM, PLMS, Euler-ancestral), classifier-free guidance, and a 2D U-Net (`Unet2d`) with ResBlocks, spatial transformer layers, and timestep embeddings. The `DiffusionModel` ties schedules, sampling, and the U-Net together with an end-to-end `sample()` API, and `DiffusionBuilder` provides fluent configuration.

## Features

- **Schedules**: `LinearSchedule`, `CosineSchedule`, `ScaledLinearSchedule` — all implementing `NoiseSchedule`; `DiffusionConfig` defaults to 1000 timesteps.
- **Samplers**: `DdpmSampler`, `DdimSampler` (configurable `eta`), `PlmsSampler`, `EulerAncestralSampler`.
- **Guidance**: `GuidanceConfig` for classifier-free guidance (CFG) mixing and dynamic thresholding.
- **U-Net**: `Unet2d` with `UnetConfig`, ResBlocks, spatial transformers (cross-attention), and timestep embeddings.
- **Training**: `diffusion/` submodule with loss routines and training helpers; `DiffusionState` for pipeline state.
- **Model**: `DiffusionModel::new(config)` + `sample(shape) -> Tensor`; `DiffusionBuilder` for fluent setup.
- **Supporting**: `ConditioningContext` (text/class/image conditioning), `IdentityLatentCodec` (latent-space adapter), `EvalReport` (step tracking), and utility ops.

## Modules

| Module | Contents |
|---|---|
| `schedules/` | `linear`, `cosine`, `scaled` (`NoiseSchedule` trait) |
| `samplers/` | `ddpm`, `ddim`, `plms`, `ancestral` |
| `unet/` | `blocks` (ResBlocks/transformers), `embeddings`, `sampling_layers` (`Unet2d`) |
| `diffusion/` | `losses`, `train`, `mod` (`DiffusionModel`) |
| `guidance.rs` | `GuidanceConfig` (CFG) |
| `conditioning.rs` | `ConditioningContext` |
| `latent.rs` | `LatentConfig`, `IdentityLatentCodec` |
| `eval.rs` | `EvalReport` |
| `builder.rs`, `config.rs`, `core.rs`, `ops.rs`, `utils.rs`, `impl.rs` | `DiffusionBuilder`, `DiffusionConfig`, `DiffusionState`, ops, RNG/utilities |

## Quick Start

```rust
use brain_diffusion::prelude::*;

let cfg = DiffusionConfig::default();
assert_eq!(cfg.timesteps, 1000);

let model = DiffusionModel::new(cfg);
let sample = model.sample(&[1, 3, 32, 32]);
assert_eq!(sample.shape(), &[1, 3, 32, 32]);
```

## Testing

```bash
cargo test -p brain-diffusion -j 2
```

## Workspace Role

Generative diffusion model family for the Brain ecosystem. Depends only on `brain-core` (tensors) — zero external dependencies, 100% safe Rust.