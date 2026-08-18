# Brain Deep Learning Framework

> A next-generation deep learning framework built in Rust that surpasses PyTorch and TensorFlow
> in architecture design, algorithm efficiency, memory safety, and developer experience.

## Overview

Brain is a comprehensive, production-grade deep learning framework built in 100% safe, pure, dependency-free Rust. Every algorithm is implemented from first principles with zero external BLAS, LAPACK, or C++ dependencies.

## Key Features in Brain 1.0

- **Reverse-Mode Autograd with Verified VJPs**: Analytical Vector-Jacobian Products for Linear, Conv2d, ConvTranspose2d, MaxPool2d, AvgPool2d, Embedding, and elementwise operations, verified against central finite differences.
- **Deep Graph Memory Safety**: Bounded memory via `Tape::drain` and iterative deconstruction (`take_parents`), handling 100,000+ node chains without call-stack overflow.
- **Cache-Blocked GEMM**: Pure-Rust 64x64 cache-tiled matrix multiplication in `brain-core` delivering high throughput with zero dependencies.
- **Closed-Form Validated Optimizers**: Exact 1-step analytical verification for SGD, Adam, AdamW (decoupled weight decay), and learning rate schedulers (`StepLR`, `CosineAnnealingLR`).
- **Complete Module Stack**: `Linear`, `Conv2d`, `BatchNorm2d` (with running statistics tracking), `LayerNorm`, `Embedding`, `Sequential`, `Trainer`.
- **Ecosystem & Quantization**:
  - `brain-onnx`: Pure-Rust ONNX protobuf parser, IR lowering, graph verification, and interpreter (opset 17).
  - `brain-quantization`: Dynamic 8-bit integer quantization and unstructured magnitude pruning.
- **Developer CLI**: Full toolchain for model development (`brain make`, `brain check`, `brain run`, `brain train`).

---

## Quick Start (CLI)

```bash
# Build the Brain CLI
cargo build -p brain --release

# 1. Train and checkpoint a model from a dataset
brain make my_model.brain --data sample_data.csv --arch convnet --epochs 20 --lr 0.1

# 2. Inspect checkpoint parameters and verify health
brain check my_model.brain

# 3. Run inference on full dataset or single input sample
brain run my_model.brain --data sample_data.csv
brain run my_model.brain --input "1.0, -2.5"
```

## Quick Start (Rust Code)

```rust
use brain_core::Tensor;
use brain_train::{Batch, Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential, Trainer};

fn main() {
    // 1. Define model architecture
    let model = Sequential::new()
        .add(Conv2d::new(1, 4, 3, true))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Flatten::new())
        .add(Linear::new(4 * 3 * 3, 2, true));

    // 2. Build trainer
    let mut trainer = Trainer::builder()
        .model(model)
        .learning_rate(0.1)
        .build()
        .unwrap();

    // 3. Train on mini-batches
    let inputs = Tensor::from_vec(vec![0.1; 8 * 1 * 6 * 6], vec![8, 1, 6, 6]);
    let batch = Batch::new(inputs, vec![0, 0, 0, 0, 1, 1, 1, 1]).unwrap();
    let summary = trainer.fit(&[batch], 15).unwrap();

    println!("Trained model: loss={:.4}, accuracy={:.1}%", summary.loss, summary.accuracy * 100.0);
}
```

## Documentation

- [Coverage Matrix](docs/coverage_matrix.md)
- [API Surface](docs/api_surface.md)
- [Ecosystem Crate Triage](docs/ecosystem_status.md)
- [Developer & Contributor Guide](AGENTS.md)
- [Changelog](CHANGELOG.md)

## License

MIT License