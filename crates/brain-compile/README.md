# `brain-compile` (v0.2.0)

> Graph Lowering, JIT Kernel Compilation, Operator Fusion, and Pure-Rust Execution Engine.

## Overview

`brain-compile` is an optimizing graph compiler that translates computation graphs into fused, high-efficiency execution schedules. It provides dead code elimination, constant folding, horizontal & vertical operator fusion, memory layout planning, buffer reuse, and JIT execution.

## Architecture

| Module | Description |
|---|---|
| `ir` | High-level and Low-level Intermediate Representation (IR) graphs and nodes |
| `passes` | Optimization passes: Dead code elimination, constant folding, algebraic simplification |
| `fusion` | Operator fusion engine: Conv+BatchNorm+ReLU, Gemm+Bias+Activation, element-wise chaining |
| `allocator` | Memory buffer reuse planner minimizing total peak allocations during graph execution |
| `engine` | Pure Rust JIT execution runtime with compiled execution plans |

## Quality & Verification

- **Tests**: 14,077 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-compile -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
