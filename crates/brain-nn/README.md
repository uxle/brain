# `brain-nn`

Pure-Rust neural network layer library: modules, activations, layers, normalization, initialization, and pruning.

## Overview

`brain-nn` provides the `Module` trait and a broad catalog of tensor-only layers for the Brain framework: dense/convolutional/recurrent layers, attention, 30+ activation functions, normalization, dropout, parameter initialization schemes, and magnitude/structured pruning. All computation runs on `brain-core` tensors in 100% safe Rust.

## Features

- **Module system** — `Module` trait (`forward`, `parameters`), `Parameter`, `Buffer`, `NamedParameter`, `ModuleDict`, `ModuleList`.
- **Layers** — `Linear`, `Conv2d`, `ConvTranspose2d`, `Embedding`, `MultiheadAttention`, `LSTM`, `GRU`, `Bilinear`, `Identity`, `AvgPool2d`, `MaxPool2d`, `AdaptiveAvgPool2d`, `AdaptiveMaxPool2d`, `PixelShuffle`.
- **Activations** — `relu`, `leaky_relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`, `gelu`, `fast_gelu`, `silu`/`swish`, `mish`, `elu`, `celu`, `selu`, `glu`, `swiglu`, `hard_sigmoid`, `hard_swish`, `softplus`, `softsign`, plus the `extra` family: `prelu`, `log_sigmoid`, `tanh_shrink`, `hard_shrink`, `soft_shrink`, `shrink`, `thresholded_relu`, `threshold`, `relu6`, `softmin`, `quiet_softmax`.
- **Normalization** — `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `InstanceNorm2d`.
- **Dropout** — `Dropout`, `AlphaDropout`, `Dropout2d`, `FusedDropout`.
- **Initialization** — Kaiming (He), Xavier (Glorot), Uniform, Normal, Orthogonal, scaled residual, and `zero_init_last_layer` schedules via `InitConfig` / `InitPolicy`.
- **Containers & hooks** — `Sequential`, `SequentialNamed`, `NamedModule`, `HookRegistry` with forward pre/post hooks.
- **Pruning** — `PruningMask` for magnitude and structured pruning.

## Modules

| Module | Description |
|---|---|
| `module` | `Module` trait, `Parameter`, `Buffer`, `NamedParameter`, `ModuleDict`, `ModuleList` |
| `layers` | Linear, conv, recurrent, attention, embedding, pooling, pixel-shuffle layers |
| `activations` | 30+ activation functions and stateful activation structs |
| `normalization` | Batch/Instance/Group/Layer/RMS normalization |
| `init` | Kaiming, Xavier, orthogonal and schedule-based parameter init |
| `dropout` | Standard, alpha, 2d and fused dropout |
| `containers` | `Sequential`, `SequentialNamed`, `NamedModule` |
| `hooks` | Forward pre/post execution hook registry |
| `pruning` | Magnitude and structured `PruningMask` |

## Quick Start

```rust
use brain_core::Tensor;
use brain_nn::{relu, Linear};

let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
let layer = Linear::new(3, 2, true);          // in_features, out_features, bias
let out = layer.forward(&x).unwrap();          // [1, 2]
let activated = relu(&out);
println!("{:?}", activated.to_vec());
```

## Testing

```bash
cargo test -p brain-nn --test activations_test -j 2
cargo test -p brain-nn --test layer_grad_check -j 2
cargo test -p brain-nn -j 2
```

## Workspace Role

Depends on `brain-core`. Consumers: `brain-train`, `brain-rl`, and the `brain` facade (via its `nn` feature).
