# AGENTS.md — Developer & Contributor Guide

Welcome to the **Brain** framework codebase. This document outlines architectural invariants, development workflows, testing rules, and resource safety directives for human contributors and AI agents alike.

---

## 1. System Invariants & Resource Guidelines

> [!CAUTION]
> **Strict Concurrency Rule**: When invoking `cargo build` or `cargo test`, **ALWAYS** pass `-j 2` (or `-j 1`). Never invoke unconstrained full-workspace builds (`cargo test --workspace` across all 33 crates simultaneously) to avoid CPU/memory starvation on developer host machines.

Always target tests per crate:
```bash
cargo test -p brain-core --test numerical_check -j 2
cargo test -p brain-autograd --test grad_check -j 2
cargo test -p brain-train --test trainer_regression -j 2
cargo test -p brain-onnx --test onnx_roundtrip -j 2
cargo test -p brain-quantization --test quant_linear -j 2
```

---

## 2. Codebase Map

- **`crates/brain-core`**: Tensor N-D array primitives, memory allocators, shape algebra, cache-blocked GEMM, BLAS L1-3, linalg (LU/QR/Cholesky/SVD), FFT, pooling (incl. adaptive), reductions (incl. cumsum/cumprod/var_mean), `BrainMind` (chatbot + neural SGD/Adam training).
- **`crates/brain-autograd`**: Reverse-mode automatic differentiation engine (`Value`, `GradFn`, `Tape`) with ~30 verified ops (abs, clamp, sin, cos, recip, square, sign, min_elem, max_elem, where_cond, ...), plus Hessian/Jacobian/JVP and checkpointing.
- **`crates/brain-nn`**: Neural network layers (`Linear`, `Conv1d/2d`, `ConvTranspose2d`, `LSTM`, `GRU`, `MultiheadAttention`, `PixelShuffle`, adaptive pools), 30+ activations (`PReLU`, `LogSigmoid`, `QuietSoftmax`, `ReLU6`, ...), normalization (`BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `InstanceNorm2d`).
- **`crates/brain-loss`**: 30+ loss functions with differentiable `forward_value` methods (CE, focal, KLDiv, Huber, Quantile, InfoNCE, SimCLR, Triplet, ArcFace, CEDice, distillation, ...).
- **`crates/brain-optim`**: 16 optimizers (SGD, Adam/AdamW/Adamax/Nadam, RAdam, Lamb, Lion, NovoGrad, RMSprop, Adagrad, Adadelta), 12 LR schedulers, clipping, `StateDict`.
- **`crates/brain-train`**: Trainer abstraction, mini-batching, `ModelState` checkpoints, callbacks (EarlyStopping, MetricHistoryLogger).
- **`crates/brain-metric`**: 60+ evaluation metrics (accuracy, ROC/PR AUC, MCC, perplexity, MRR, NDCG, mAP, IoU, calibration, reports).
- **`crates/brain-transformer`**: Transformer encoder/decoder, MHA/GQA/MQA/Cross/Relative/Flash attention, RoPE, Alibi, KV cache, Llama/GPT/T5/Bert lites, generation pipelines.
- **`crates/brain-rnn`**: LSTM/GRU/Vanilla/Peephole cells and sequences, Bidirectional, PackedSequence, BeamSearch, TeacherForcing.
- **`crates/brain-onnx`**: Pure-Rust ONNX protobuf parser, IR lowering, and interpreter (opset 9-21).
- **`crates/brain-quantization`**: Dynamic/static Int8 quantization, calibration, QLinear/QConv2d, pruning, CSR sparse ops.
- **`crates/brain-graph` / `brain-compile`**: Static graph IR with passes, scheduling, interpreter, JIT cache, memory plans.
- **`crates/brain-data` / `brain-dataset`**: DataSource, streaming/mmap loaders, DataLoader + WorkerPool, transforms, samplers, splits.
- **`crates/brain-cli` & `crates/brain`**: CLI application binary (`brain make`, `brain check`, `brain run`, `brain train`, `brain chat`, `brain dataset`, `brain doctor`, ...) and umbrella facade.
- **Research**: `brain-rl` (DQN/PPO/A2C/SAC), `brain-gnn` (GCN/GAT/SAGE), `brain-diffusion` (DDPM/DDIM/PLMS), `brain-gan`, `brain-neuroevolution` (GA/CMA-ES/HyperNEAT), `brain-vit`, `brain-cv`, `brain-audio`, `brain-text`.
- **Systems**: `brain-distributed` (ring/tree allreduce, data/model/tensor/pipeline parallelism), `brain-federated` (FedAvg, secure aggregation), `brain-benchmark`, `brain-export` (ONNX/TFLite/CoreML/WebNN), `brain-utils`.

---

## 3. Correctness Protocol for Adding New Operators

When adding a new neural operation to `brain-core` / `brain-autograd`:
1. Implement forward math with checked shape calculation.
2. Implement exact analytical Vector-Jacobian Product (VJP) in `crates/brain-autograd/src/ops/`.
3. Wire the operator into `GradFn`, `parents()`, `take_parents()`, and `apply_vjp()`.
4. Add a finite-difference verification test in `crates/brain-autograd/tests/grad_check.rs` asserting numeric vs analytical difference $< 1e-4$.
5. Never return dummy zero gradients for trainable operations.

---

## 4. Useful Verification Commands

- **Run Core Tests**: `./scripts/ci.sh`
- **Build CLI**: `cargo build -p brain -j 2`
- **Run Examples**: `cargo run --example convnet_train -j 2`
