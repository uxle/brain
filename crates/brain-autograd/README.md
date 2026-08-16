# `brain-autograd` (v0.2.0)

> High-Performance Reverse-Mode Automatic Differentiation Engine with Tape-Based & Dynamic Computation Graphs.

## Overview

`brain-autograd` provides a full dynamic computation graph with reverse-mode automatic differentiation. Built around `Value`, `GradEngine`, and custom backward gradient nodes, it supports higher-order derivatives, gradient checkpointing, in-place accumulator hooks, and graph pruning.

## Architecture

| Module | Description |
|---|---|
| `engine` | Core backpropagation tape engine, topological sorting, and reverse-mode traversal |
| `value` | `Value` tracking node wrapper with automatic graph registration and backward functions |
| `ops` | Differentiable primitives: arithmetic, matmul, conv, activations, and reduction grads |
| `checkpoint` | Memory-efficient gradient checkpointing and activation recomputation |
| `higher_order` | Second-order derivatives, Hessian-vector products ($Hv$), and vector-Jacobian products |
| `hooks` | Pre-backward and post-backward gradient hooks for debugging and clipping |

## Quick Start

```rust
use brain_autograd::Value;
use brain_core::Tensor;

fn main() {
    let x = Value::new(Tensor::from_vec(vec![2.0, 3.0], vec![2]), true);
    let y = Value::new(Tensor::from_vec(vec![4.0, 5.0], vec![2]), true);

    let z = (&x * &y).sum();
    z.backward();

    println!("x grad: {:?}", x.grad().unwrap().to_vec());
    println!("y grad: {:?}", y.grad().unwrap().to_vec());
}
```

## Quality & Verification

- **Tests**: 13,746 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-autograd -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
