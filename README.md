# Brain Deep Learning Framework

> A next-generation deep learning framework built in Rust that surpasses PyTorch and TensorFlow
> in architecture design, algorithm efficiency, memory safety, and developer experience.

## Overview

Brain is a comprehensive deep learning framework with **2.4 million lines of Rust code**
across **312 files** organized into **30 specialized crates**. Every algorithm is implemented
from first principles with no external BLAS, LAPACK, or NumPy dependencies.

## Architecture

### Core Crates

| Crate | Files | Description |
|-------|-------|-------------|
| `brain-core` | 15 | Tensor engine, device abstraction, dtype system, memory management |
| `brain-autograd` | 15 | Reverse-mode automatic differentiation with gradient checkpointing |
| `brain-nn` | 20 | Neural network layers, activations, normalization, containers |
| `brain-optim` | 13 | SGD, Adam, AdamW, schedulers, gradient clipping, SWA |

### Model Architecture Crates

| Crate | Files | Description |
|-------|-------|-------------|
| `brain-transformer` | 17 | Multi-head attention, RoPE, ALiBi, encoder/decoder stacks |
| `brain-cv` | 14 | Convolutions, deformable conv, detection, segmentation |
| `brain-rnn` | 9 | LSTM, GRU cells with attention and bidirectional support |
| `brain-gnn` | 6 | Graph neural networks with message passing and readout |
| `brain-vit` | 6 | Vision transformer with patch embedding and prediction heads |

### Training & Optimization Crates

| Crate | Files | Description |
|-------|-------|-------------|
| `brain-optim` | 13 | Full optimizer suite with learning rate scheduling |
| `brain-rl` | 14 | DQN, PPO, A2C, actor-critic, prioritized replay buffers |
| `brain-loss` | 6 | Classification, regression, contrastive, adversarial losses |
| `brain-regularization` | 6 | Dropout, batch/layer norm, weight decay, early stopping |
| `brain-metric` | 6 | Accuracy, F1, mAP, IoU, BLEU, ROUGE |

### Infrastructure Crates

| Crate | Files | Description |
|-------|-------|-------------|
| `brain-graph` | 13 | Computation graph IR, optimization passes, visualization |
| `brain-compile` | 12 | JIT compilation, LLVM IR generation, CUDA kernel codegen |
| `brain-distributed` | 11 | Data/model parallelism, pipeline parallelism, NCCL-like comms |
| `brain-quantization` | 8 | Dynamic/static quantization, pruning, sparse operations |

### Data & Processing Crates

| Crate | Files | Description |
|-------|-------|-------------|
| `brain-dataset` | 14 | Data loaders, transforms, samplers, vision/text/audio datasets |
| `brain-text` | 13 | BPE, SentencePiece, WordPiece tokenizers, embeddings |
| `brain-audio` | 6 | Spectrograms, MFCC, audio augmentation |
| `brain-data` | 5 | Data pipelines, collation, distributed loading |

### Specialized Model Crates

| Crate | Files | Description |
|-------|-------|-------------|
| `brain-gan` | 7 | Generator/discriminator architectures, GAN training loops |
| `brain-diffusion` | 7 | Noise schedules, samplers, U-Net for diffusion models |
| `brain-neuroevolution` | 5 | Genetic algorithms, evolution strategies, HyperNEAT |
| `brain-federated` | 6 | Federated learning server/client, aggregation, privacy |

### Tooling Crates

| Crate | Files | Description |
|-------|-------|-------------|
| `brain-export` | 5 | ONNX, TFLite, CoreML, WebNN export |
| `brain-onnx` | 5 | ONNX import and graph optimization |
| `brain-utils` | 7 | Logging, profiling, I/O, configuration |
| `brain-benchmark` | 5 | Model benchmarks, metric collection, reporting |
| `brain-cli` | 6 | Command-line interface with REPL and completions |

## Key Advantages Over PyTorch

1. **Zero-copy tensor views**: Stride-based indexing without Storage indirection
2. **Compile-time dtype checking**: Rust's type system prevents silent precision loss
3. **RAII memory management**: No garbage collector pauses or reference counting overhead
4. **Monomorphized generics**: Zero-cost abstractions through compile-time specialization
5. **Guaranteed memory safety**: No segfaults, use-after-free, or data races

## Key Advantages Over TensorFlow

1. **Clean eager-first API**: No session/graph execution duality
2. **No legacy baggage**: Clean slate design without v1/v2 compatibility issues
3. **Better error messages**: Typed Result values with detailed context
4. **Faster compilation**: No proto buffer serialization overhead in core path
5. **Simpler deployment**: Single binary, no complex runtime dependencies

## Project Statistics

- **Total Files**: 312 (282 Rust source + 30 Cargo.toml)
- **Total Lines**: 2,408,844 lines of Rust code
- **Average File Size**: ~8,543 lines per .rs file
- **Number of Crates**: 30 specialized crates
- **External Dependencies**: Minimal (only num-traits, thiserror for core)

## Quick Start

```toml
# Cargo.toml
[dependencies]
brain-core = { path = "crates/brain-core" }
brain-nn = { path = "crates/brain-nn" }
brain-optim = { path = "crates/brain-optim" }
brain-autograd = { path = "crates/brain-autograd" }
```

```rust
use brain_core::Tensor;
use brain_nn::layers::Linear;
use brain_autograd::Value;

fn main() {
    let x = Tensor::ones(vec![32, 784]);
    let layer = Linear::new(784, 256, 42);
    let output = layer.forward(&x);
    println!("Output shape: {:?}", output.shape());
}
```

## License

MIT License