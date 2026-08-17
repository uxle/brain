# `brain-graph` (v0.2.0)

> Production-Grade Computation-Graph IR, Optimization Passes, Static Analysis, Interpretation, Execution Scheduling, and Graphviz/JSON Export.

## Overview

`brain-graph` delivers a comprehensive intermediate representation (IR) and compiler optimization pipeline for deep learning computation graphs. Built with pure, safe Rust and zero external dependencies, it provides full structural and semantic verification, forward shape inference, a suite of graph optimization passes (constant folding, dead code elimination, common subexpression elimination, operator fusion, layout transforms, in-place buffer planning), multi-stage execution scheduling, graph diffing, memory profiling, reference interpretation against `brain-core` tensors, and Graphviz DOT / JSON serialization.

## Architecture

| Module | Description |
|---|---|
| `ir` | `GraphIr`, `GraphNode`, `GraphEdge`, `GraphValue`, `OpKind`, verification, shape inference |
| `passes` | Optimization passes: constant folding, dead code elimination, CSE, fusion, layout, in-place |
| `topology` | Topological sorting (Kahn, DFS), node rank assignment, and critical path analysis |
| `schedule` | Stage-based execution scheduling and parallel region extraction |
| `dot` | Graphviz DOT format exporter with styled operators and memory annotations |
| `json` | Deterministic JSON graph serialization and deserialization |
| `clone` | Subgraph extraction and deep cloning with complete ID remapping |
| `diff` | Structural and semantic graph diffing and equivalence checks |
| `interp` | Pure Rust reference interpreter executing against `brain-core` tensors |
| `profile` | Memory liveness tracking, peak memory estimation, and FLOP calculation |
| `analyze` | Cycle detection, parallelism factor, and fusion opportunity mining |
| `compute` | Arithmetic intensity and computational cost modeling |
| `optimize` | High-level optimization coordinator (`optimize(graph, level)`) |
| `transform` | Algebraic simplification rules ($x \cdot 1 \to x$, $x + 0 \to x$, $x - x \to 0$) |
| `helper` | Ready-to-use demo architectures: MLP, CNN, Transformer block |
| `builder` | Fluent `GraphBuilder` incremental construction API |
| `config` | `GraphConfig`, `OptLevel` (O0–O3), `VerificationLevel` |
| `core` | `NodeId`, `ValueId`, `EdgeId`, `Shape`, `DType`, `DeviceKind`, `GraphError` |
| `ops` | Graph operator construction functions and direct tensor execution |

## Quick Start

```rust
use brain_graph::{GraphBuilder, OptLevel, optimize, to_dot, to_json, DType, OpKind};

fn main() {
    let mut builder = GraphBuilder::new("mlp_block");
    let x = builder.add_input("x", vec![1, 128], DType::F32);
    let w1 = builder.add_constant("w1", vec![128, 64], vec![0.01; 128 * 64]);
    let h1 = builder.add_node("mm1", OpKind::MatMul, vec![x, w1], vec![1, 64]);
    let act = builder.add_node("relu1", OpKind::Relu, vec![h1], vec![1, 64]);
    builder.mark_output(act);

    let mut graph = builder.build().unwrap();

    // Optimize graph
    let report = optimize(&mut graph, OptLevel::O2).unwrap();
    println!("Passes applied: {}, Final nodes: {}", report.passes_applied, report.final_nodes);

    // Export to Graphviz DOT
    let dot_str = to_dot(&graph);
    println!("{}", dot_str);
}
```

## Quality & Verification

- **Tests**: 7,688 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-graph -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
