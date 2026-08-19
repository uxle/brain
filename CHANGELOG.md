# Changelog

All notable changes to the Brain framework will be documented in this file.

## [Unreleased]

### Added
- **`brain-autograd`**: 10 new operators with analytical VJPs verified against central finite differences: `abs`, `clamp`, `sin`, `cos`, `recip`, `square`, `sign` (unary) and `min_elem`, `max_elem` (tie-split 0.5/0.5), `where_cond` (ternary mask). All wired into `GradFn` (`parents`, `take_parents`, `op_name`, `apply_vjp`) with panic-asserting `grad_check` tests.
- **`brain-core` tensor ops**: `matvec`, `cosine_similarity` (dim-wise), `var_mean`, `var_along_dim`, `std_along_dim`, `cumsum`, `cumprod`, `repeat` (torch `repeat` semantics), `adaptive_avg_pool2d`, `adaptive_max_pool2d` — all with unit tests.
- **`brain-nn` activations** (new `activations/extra.rs`): `PReLU`, `LogSigmoid` (stable), `TanhShrink`, `HardShrink`, `SoftShrink`, `Shrink`, `ThresholdedReLU`, `Threshold`, `ReLU6`, `Softmin`, `QuietSoftmax` (DeepSeek-R1 per-element temperature) — struct wrappers + free functions.
- **`brain-nn` layers**: `AdaptiveAvgPool2d`, `AdaptiveMaxPool2d`, `PixelShuffle` (sub-pixel conv), `InstanceNorm2d` (per-sample-channel, affine/no-affine).
- **Chatbot truthfulness remediation** (`brain-core::BrainMind`): colon-fact indexing gate in `teach_file` (`END` bounding, non-indented `KEY: VALUE` only), subject-targeted `question_subject`/`search_knowledge_facts`, question gate in `talk()` (unknown questions fall back to neural continuation or honest "I don't know"), `learn_sentence` stop subjects.
- **Real neural training in `BrainMind`**: exact cross-entropy backpropagation (`neural_backward`) through LM head, RMSNorm, Swish FFN, causal self-attention, and embeddings; `neural_train_sequence` (SGD) and `neural_adam_train_sequence` (Adam, bias-corrected, moments persisted) replace the former constant-bump trainer; Adam integrated into `talk()` and `teach_file`. Gradients verified numerically (all 8 weight matrices ≤ ~1e-5 error).

### Changed
- `brain-autograd` grad_check suite now contains real panic-asserting tests for every new op (28 passed, 3 deferred).

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
