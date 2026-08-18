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

- **`crates/brain-core`**: Tensor N-D array primitives, memory allocators, shape algebra, and cache-blocked GEMM.
- **`crates/brain-autograd`**: Reverse-mode automatic differentiation engine (`Value`, `GradFn`, `Tape`).
- **`crates/brain-nn`**: Neural network layers (`Linear`, `Conv2d`, `BatchNorm2d`, `Embedding`, `Sequential`).
- **`crates/brain-loss`**: Loss functions with differentiable `forward_value` methods (`CrossEntropyLoss`, `MSELoss`, `SmoothL1Loss`).
- **`crates/brain-optim`**: Numerical optimizers (`Sgd`, `Adam`, `AdamW`), LR schedulers, and `StateDict`.
- **`crates/brain-train`**: Trainer abstraction, mini-batching, and checkpointing.
- **`crates/brain-onnx`**: Pure-Rust ONNX protobuf parser, IR lowering, and interpreter.
- **`crates/brain-quantization`**: Dynamic Int8 quantization and magnitude pruning.
- **`crates/brain-cli` & `crates/brain`**: CLI application binary (`brain make`, `brain check`, `brain run`, `brain train`).

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
