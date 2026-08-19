# `brain-graph`

Pure-Rust computation-graph IR with verification, optimization passes, interpretation, scheduling, profiling, and DOT/JSON export.

## Overview

`brain-graph` is a compiler-style front end for neural computation graphs built on `brain-core` tensors with zero external dependencies. It offers a typed IR (`GraphIr`) with shape inference and structural verification, a pass pipeline (constant folding, DCE, CSE, fusion, layout, in-place planning), a reference `GraphInterpreter`, topological scheduling, memory/FLOP profiling, graph diffing, and Graphviz/JSON serialization.

## Features

- **IR & verification**: `GraphIr`/`GraphNode`/`GraphEdge`/`GraphValue`, `OpKind`/`OpRegistry`, `verify_graph`, `infer_graph_shapes`, `process_with_verification`.
- **Optimization passes**: `fold_constants`, `eliminate_dead_code`, `eliminate_cse`, `eliminate_layout_transforms`, `plan_fusion`, `plan_inplace_operations`, coordinated by `PassManager` and the top-level `optimize(graph, level)` with `OptimizeReport`.
- **Execution**: `GraphInterpreter` and `run_graph` evaluate graphs against `brain-core` tensors.
- **Analysis**: topological order (`compute_topological_order`), cycle/fusion/parallelism analysis (`analyze_*`), cost modeling (`compute_costs`), and memory profiling (`profile_graph`).
- **Export & diffing**: `to_dot` (styled Graphviz), `to_json` (deterministic), `diff_graphs`/`GraphDiff`, `clone_subgraph`.
- **Ready-made architectures**: `build_mlp_graph`, `build_cnn_graph`, `build_transformer_graph`.
- **Fluent construction**: `GraphBuilder` incremental API with `GraphConfig`, `OptLevel`, and `VerificationLevel`.

## Modules

| Module | Contents |
|---|---|
| `ir` | `GraphIr`, `GraphNode`, `GraphEdge`, `GraphValue`, `OpKind`, `OpRegistry`, verification, shape inference |
| `passes` | constant folding, DCE, CSE, fusion, layout, in-place; `GraphPass`, `PassManager`, plans |
| `topology`/`schedule` | topo sort, node rank, stage-based scheduling |
| `interp`/`impl_` | reference interpreter, `run_graph`, memory estimation |
| `analyze`/`compute`/`profile` | cycle detection, parallelism, costs, memory/FLOP profiling |
| `dot`/`json`/`clone`/`diff` | Graphviz export, JSON serialization, subgraph cloning, diffing |
| `optimize`/`transform` | optimization coordinator, algebraic rewrites |
| `helper`/`builder`/`config`/`core`/`ops` | demo graphs, `GraphBuilder`, configs, IDs/shapes/dtypes, op constructors |

## Quick Start

```rust
use brain_core::Tensor;
use brain_graph::builder::GraphBuilder;
use brain_graph::core::DType;
use brain_graph::interp::GraphInterpreter;
use brain_graph::ir::ops::OpKind;

let mut b = GraphBuilder::new("linear_relu");
let x = b.add_input("x", vec![2, 2], DType::F32);
let w = b.add_constant("w", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
let mm = b.add_node("matmul", OpKind::MatMul, vec![x, w], vec![2, 2]);
b.mark_output(mm);
let graph = b.build().unwrap();

let mut interp = GraphInterpreter::new();
let out = interp.run(&graph, &[Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])]).unwrap();
```

## Testing

```bash
cargo test -p brain-graph -j 2
```

## Workspace Role

Depends solely on `brain-core`; `brain-graph` serves as the framework's graph compiler/optimizer layer, enabling graph-level optimization and interpretation for training and inference tooling.