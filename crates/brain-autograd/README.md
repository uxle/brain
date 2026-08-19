# `brain-autograd`

Pure-Rust reverse-mode automatic differentiation engine with 30+ differentiable operators.

## Overview

`brain-autograd` builds a compute graph of `Value` nodes, each carrying a `GradFn` that knows its parents and applies the exact analytical Vector-Jacobian Product (VJP) during `backward()`. It adds gradient checkpointing, parallel reverse sweeps, a recording `Tape` for op-level inspection, and higher-order functional transforms (`grad`, `jacobian`, `hessian`, `vjp`, `jvp`). Built on `brain-core` tensors.

## Features

- **Value-based autograd** — `Value::scalar`, `Value::from_slice`, `Value::new`, `detach`, `set_requires_grad`, `zero_grad`, `accumulate_grad`, `backward`, `backward_with_grad`.
- **30+ differentiable operators** — arithmetic (`add`, `sub`, `mul`, `div`, `pow`, `matmul`), unary (`exp`, `log`, `sqrt`, `relu`, `sigmoid`, `tanh`, `abs`, `clamp`, `sin`, `cos`, `recip`, `square`, `sign`), reductions (`sum`, `mean`), elementwise pair ops (`min_elem`, `max_elem`, `where_cond`), `softmax` / `log_softmax`, and neural ops (`conv2d`, `conv_transpose2d`, `avg_pool2d`, `max_pool2d`).
- **Exact VJPs** — gradient rules in `ops/` (unary, binary, reduction, conv, pool, fft, index, linalg, quant, sparse) verified against finite differences.
- **Functional transforms** — `grad`, `grad_and_hess`, `hessian`, `jacobian`, `jvp`, `value_and_grad`, `vjp` (`graph_closure`).
- **Gradient checkpointing** — selective `checkpoint` with `CheckpointPolicy`, full-graph `RecomputeGraph`, and `CpuOffloader` for memory-bounded training.
- **Parallel & mixed precision** — topological DAG levels (`compute_dag_levels`) driving `parallel_backward`; `GradScaler` for loss scaling / grad unscaling.
- **Op recording** — `Tape` with `start_recording` / `stop_recording` / `with_tape` for `OpRecord` inspection and memory-bounded tapes.

## Modules

| Module | Description |
|---|---|
| `value` | `Value` node with data, grad, `GradFn`, and chained op methods |
| `ops` | Differentiable forward ops + VJP gradient rules |
| `grad_fns` | `GradFn` enum and VJP implementations per op family |
| `backward` | `backward_from`, `topological_sort`, DAG level computation |
| `graph_closure` | `grad`, `jacobian`, `hessian`, `vjp`, `jvp` functional transforms |
| `tape` | `OpRecord`, `Tape`, recording lifecycle helpers |
| `checkpoint` | Selective / recompute / CPU-offload gradient checkpointing |
| `engine` | `parallel_backward` and mixed-precision `GradScaler` |

## Quick Start

```rust
use brain_autograd::Value;

let mut x = Value::scalar(3.0);
x.set_requires_grad(true);
let y = x.mul(&x);
y.backward().unwrap();
assert_eq!(x.grad().unwrap().get(0), 6.0); // dy/dx = 2x

// Higher-order transform:
let x2 = Value::scalar(4.0);
let g = brain_autograd::grad(|v| v.mul(v), &x2).unwrap().unwrap();
assert_eq!(g.get(0), 8.0);
```

## Testing

```bash
cargo test -p brain-autograd --test grad_check -j 2
cargo test -p brain-autograd -j 2
```

`grad_check` asserts analytic vs finite-difference gradients agree within `1e-4` for every operator.

## Workspace Role

Depends on `brain-core`. Consumers: `brain-loss`, `brain-train`, `brain-rl`, `brain-rnn`, `brain-text`, `brain-cli`, and the `brain` facade.
