# `brain-compile`

> SSA-style IR graph, optimization passes, JIT cache, and a pure-Rust interpreter execution engine.

## Overview

`brain-compile` provides a typed intermediate representation (`IrGraph` with `IrValue`/`IrOp` nodes), a pass pipeline over that IR, and an `Interpreter` backend that evaluates graphs on `brain-core` tensors. It adds compilation support infrastructure: a JIT cache keyed by compiled graphs, memory planning, scheduling, FLOP analysis, backend codegen stubs for CUDA/LLVM, and DOT/text IR export for debugging.

## Features

- Typed IR: `IrType` (e.g. `F64`), `IrValue`, `IrOp`, and `IrGraph` with `add_value`/`add_node`, inputs and outputs
- `Pass` trait and `PassManager` with broadcast, dead-code elimination, constant folding, fusion, and layout passes; `PassManager::from_options` wired to `CompileOptions`
- `Interpreter` backend executing IR graphs against `brain-core::Tensor` (verified numerically in tests)
- `JitCache::get_or_compile` caching compiled graphs per `CompileOptions`
- `MemoryPlan::create_plan` and `SchedulePlan::compute_schedule` for execution planning; `analyze_tensor_lifetimes` lifetime analysis
- `estimate_total_flops`, `OpCostInfo` with `arithmetic_intensity`, `apply_algebraic_rewrites` transforms
- `export_dot` / `export_text` IR dumpers and `generate_cuda_kernel` / `generate_llvm_ir` backend codegen entry points

## Modules

| Module | Description |
|---|---|
| `ir` | `IrType`, `IrValue`, `IrOp`, `IrGraph`, `OpKind`, IR verification |
| `builder` | `IrGraphBuilder` with checkpoint/rollback and `finish` |
| `passes` | `Pass` trait, `PassManager`, broadcast/DCE/fold/fusion/layout passes |
| `backend` | `Interpreter` (evaluates graphs), CUDA/LLVM codegen stubs, scalar/tensor helpers |
| `exec` | `ExecutionEngine::run` over a graph |
| `jit` | `JitCache::get_or_compile` |
| `plan` | `MemoryPlan::create_plan` buffer planning |
| `schedule` | `SchedulePlan::compute_schedule` |
| `compute` | `analyze_tensor_lifetimes` |
| `analyze` | `estimate_total_flops` |
| `transform` | `apply_algebraic_rewrites` |
| `profiler` | `ProfileReport::profile` per-graph cost profile |
| `export_ir` | `export_dot` / `export_text` IR serialization |
| `core` / `config` | `TargetBackend`, `OptimizationLevel`, `CompileOptions`, `CompilationError`, `JitCacheConfig`, `CompilerConfig` |
| `ops` / `helper` / `utils` / `process` | Op cost info, broadcast-shape helper, FNV hashing / byte and FLOP formatting, pipeline stages |

## Quick Start

```rust
use brain_compile::backend::interp::Interpreter;
use brain_compile::ir::{IrGraph, IrType, OpKind};
use brain_core::Tensor;

let mut graph = IrGraph::new();
let x = graph.add_value(IrType::F64, vec![1, 2]);
let y = graph.add_value(IrType::F64, vec![1, 2]);
let out = graph.add_value(IrType::F64, vec![1, 2]);
graph.inputs = vec![x, y];
graph.outputs = vec![out];
graph.add_node(OpKind::Add, vec![x, y], out);

let interp = Interpreter::new();
let res = interp.evaluate(&graph, &[Tensor::from_vec(vec![1.5, -2.5], vec![1, 2]),
                                    Tensor::from_vec(vec![0.5, 3.5], vec![1, 2])])
    .expect("evaluate");
```

## Testing

```bash
cargo test -p brain-compile -j 2
```

Covers graph/interpreter equivalence and self-hosting checks (`tests/compile_equivalence.rs`, `tests/self_host.rs`).

## Workspace Role

Depends only on `brain-core`. `brain-compile` is the optimization/compilation stage of the pipeline: graphs built from tensors get lowered, optimized by passes, cached, and executed — feeding `brain-export` with validated, IR-level models.