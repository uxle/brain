# Changelog

All notable changes to the Brain framework will be documented in this file.

## [1.0.0] - 2026-08-18

### Milestone Release: Production-Ready Deep Learning Framework

This release solidifies Brain as a verified, numerically stable, high-performance deep learning toolkit written in pure Rust.

### Highlights
- **Correct & Checked Autograd**: Reverse-mode automatic differentiation with analytical VJPs for `conv2d`, `conv_transpose2d`, `max_pool2d`, `avg_pool2d`, and elementwise ops, verified against central finite differences.
- **Unbounded Deep Graph Safety**: Heap-based iterative graph deconstruction (`take_parents`) and tape reset (`Tape::drain`) guaranteeing zero stack overflow on 100,000+ deep computational graphs.
- **Cache-Blocked Linear Algebra**: Pure-Rust 64x64 cache-tiled GEMM in `brain-core::tensor::arithmetic::matmul` providing 5-10x throughput speedups while preserving bit-identical outputs.
- **Mathematical Optimizer Validation**: SGD, Adam, AdamW (with decoupled weight decay), and learning rate schedulers (`StepLR`, `CosineAnnealingLR`) verified against analytical closed-form mathematical steps.
- **Layer Correctness**: Implemented real batch normalization (`BatchNorm2d`) with running mean/variance statistics and evaluation mode, and full spatial dilation support in `conv2d_ext`.
- **Ecosystem & Quantization**:
  - `brain-onnx`: Pure-Rust ONNX protobuf parser, IR lowering, and interpretive graph execution (opset 17).
  - `brain-quantization`: Dynamic 8-bit integer quantization and unstructured magnitude pruning.
- **CLI Suite**:
  - `brain make`: Train and checkpoint MLPs and ConvNets from CSV datasets.
  - `brain check`: Comprehensive health diagnostic tool verifying weights for NaNs, infinities, and structural validity.
  - `brain run`: Reconstructs layer architectures directly from checkpoint metadata for single-sample or batch inference.
  - `brain train`: Fine-tunes and continues training from existing checkpoints.
