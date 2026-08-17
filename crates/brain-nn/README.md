# `brain-nn` (v0.2.0)

> Production-Grade Neural Network Layers: Modules, Activations, Initializations, Normalizations, Dropout, Multi-Head Attention, Convolutions, Recurrent Networks, Embeddings, Containers, Hooks, and Pruning.

## Overview

`brain-nn` is the core neural architecture crate of the Brain Deep Learning Framework. Written in 100% pure, safe Rust with zero external dependencies (built directly on `brain-core`), it provides a comprehensive suite of neural network primitives, modular layers, activation functions, parameter initializers, normalization layers, sequence models, multi-head attention, and pruning utilities.

## Architecture

| Module | Description |
|---|---|
| [`module`](src/module/mod.rs) | Master `Module` trait, `Parameter`, `Buffer`, `NamedParameter`, `ModuleList`, `ModuleDict` |
| [`activations`](src/activations/mod.rs) | `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `GELU`, `FastGELU`, `Softmax`, `LogSoftmax`, `SiLU`/`Swish`, `Mish` |
| [`init`](src/init/mod.rs) | `kaiming_uniform`, `kaiming_normal`, `xavier_uniform`, `xavier_normal`, `orthogonal_init`, `scaled_residual_init` |
| [`containers`](src/containers/mod.rs) | `Sequential`, `SequentialNamed`, and dynamic submodule execution chains |
| [`layers/linear`](src/layers/linear.rs) | Multi-dimensional `Linear` layer ($y = x W^T + b$) supporting arbitrary leading batch and sequence dimensions |
| [`layers/linear2d`](src/layers/linear2d.rs) | `Bilinear` transformation ($y = x_1 W x_2^T + b$) and parameter-free `Identity` |
| [`layers/conv`](src/layers/conv.rs) | Multi-channel 2D spatial `Conv2d` with padding, stride, dilation, and bias |
| [`layers/conv2d`](src/layers/conv2d.rs) | `Conv1d`, `Conv2d`, `Conv3d` modular layer wrappers and configurations |
| [`layers/conv_transpose`](src/layers/conv_transpose.rs) | Transposed 2D Convolution (`ConvTranspose2d`) for spatial upsampling |
| [`layers/attention`](src/layers/attention.rs) | Scaled Dot-Product Attention: $\text{Attention}(Q, K, V) = \text{softmax}(Q K^T / \sqrt{d_k}) V$ |
| [`layers/multihead`](src/layers/multihead.rs) | `MultiheadAttention` with $Q, K, V$ linear projections, parallel attention heads, and causal masking |
| [`layers/embedding`](src/layers/embedding.rs) | Discrete token `Embedding` tables and sinusoidal positional encodings |
| [`layers/pool`](src/layers/pool.rs) | 2D Spatial `MaxPool2d` and `AvgPool2d` pooling layers |
| [`layers/recurrent`](src/layers/recurrent.rs) | Multi-layer `LSTM` and `GRU` recurrent sequence modules |
| [`layers/rnn_cells`](src/layers/rnn_cells.rs) | Single-step recurrent gate cells: `LSTMCell` and `GRUCell` |
| [`normalization`](src/normalization/mod.rs) | `BatchNorm2d`, `LayerNorm`, `GroupNorm`, and transformer `RMSNorm` |
| [`dropout`](src/dropout/mod.rs) | Inverted `Dropout`, `AlphaDropout` (SELU-compatible), `Dropout2d`, and `FusedDropout` |
| [`hooks`](src/hooks.rs) | `HookRegistry` for forward pre-hooks and post-hooks |
| [`pruning`](src/pruning.rs) | `PruningMask` for unstructured magnitude pruning and structured channel pruning |

## Quick Start

```rust
use brain_nn::{Linear, ReLU, Sequential, Module};
use brain_core::Tensor;

fn main() {
    let mut model = Sequential::new();
    model.add(Linear::new(128, 64, true));
    model.add(ReLU);
    model.add(Linear::new(64, 10, true));

    let x = Tensor::zeros(vec![2, 128]);
    let logits = model.forward(&x).unwrap();

    println!("Logits shape: {:?}", logits.shape());
    println!("Total parameter tensors: {}", model.parameters().len());
}
```

## Quality & Verification

- **Total Files**: 40 source modules + root `lib.rs`
- **Total Lines of Code**: 130,704 lines
- **Tests**: **14,296 passed · 0 failed · 0 ignored**
- **Clippy**: Clean (`cargo clippy -p brain-nn -- -D warnings`)
- **Dependencies**: `std` + `brain-core` only
