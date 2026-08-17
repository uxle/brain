//! # brain-graph
//!
//! Production-grade computation-graph IR, verification, optimization passes,
//! interpreter, scheduling, profiling, diffing, and Graphviz/JSON export.
//!
//! ## Architecture
//! - [`ir`] — `GraphIr`, `GraphNode`, `GraphEdge`, `GraphValue`, `OpKind`, verification, shape inference
//! - [`passes`] — Optimization passes: constant folding, dead code elimination, CSE, fusion, layout, in-place
//! - [`topology`] — Topological sorting (Kahn, DFS), node rank, and critical path analysis
//! - [`schedule`] — Stage-based execution scheduling and parallel region extraction
//! - [`dot`] — Graphviz DOT format exporter with styled operators
//! - [`json`] — Deterministic JSON graph serialization and deserialization
//! - [`clone`] — Subgraph extraction and deep cloning with ID remapping
//! - [`diff`] — Structural and semantic graph diffing
//! - [`interp`] — Pure Rust reference interpreter executing against `brain-core` tensors
//! - [`profile`] — Memory liveness tracking, peak memory estimation, and FLOP calculation
//! - [`analyze`] — Cycle detection, parallelism factor, and fusion opportunity mining
//! - [`compute`] — Arithmetic intensity and computational cost modeling
//! - [`optimize`] — High-level optimization coordinator (`optimize(graph, level)`)
//! - [`transform`] — Algebraic simplification rules
//! - [`helper`] — Ready-to-use demo architectures: MLP, CNN, Transformer block
//! - [`builder`] — Fluent `GraphBuilder` incremental construction API
//! - [`config`] — `GraphConfig`, `OptLevel`, `VerificationLevel`
//! - [`core`] — `NodeId`, `ValueId`, `EdgeId`, `Shape`, `DType`, `DeviceKind`, `GraphError`
//! - [`ops`] — Graph operator construction functions and tensor execution

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

pub mod analyze;
pub mod builder;
pub mod clone;
pub mod compute;
pub mod config;
pub mod core;
pub mod diff;
pub mod dot;
pub mod helper;
pub mod impl_;
pub mod interp;
pub mod ir;
pub mod json;
pub mod ops;
pub mod optimize;
pub mod passes;
pub mod process;
pub mod profile;
pub mod schedule;
pub mod topology;
pub mod transform;
pub mod utils;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use analyze::{analyze_cycles, analyze_fusion_candidates, analyze_parallelism};
pub use builder::GraphBuilder;
pub use clone::clone_subgraph;
pub use compute::{compute_costs, GraphCosts};
pub use config::{GraphConfig, OptLevel, VerificationLevel};
pub use core::{DType, DeviceKind, EdgeId, GraphError, GraphMetadata, GraphResult, NodeId, Shape, ValueId};
pub use diff::{diff_graphs, GraphDiff};
pub use dot::to_dot;
pub use helper::{build_cnn_graph, build_mlp_graph, build_transformer_graph};
pub use impl_::{run_graph, total_output_memory_bytes};
pub use interp::GraphInterpreter;
pub use ir::{
    infer_graph_shapes, verify_graph, GraphEdge, GraphIr, GraphNode, GraphValue, OpKind, OpRegistry,
};
pub use json::to_json;
pub use ops::{graph_add, graph_matmul, graph_relu, op_apply};
pub use optimize::{optimize, OptimizeReport};
pub use passes::{
    eliminate_cse, eliminate_dead_code, eliminate_layout_transforms, fold_constants,
    plan_fusion, plan_inplace_operations, ConstFoldPass, CsePass, DeadCodeElimPass,
    FusionPass, FusionPlan, GraphPass, InplacePass, InplacePlan, LayoutPass, PassManager,
};
pub use process::process_with_verification;
pub use profile::{profile_graph, GraphProfile};
pub use schedule::{generate_schedule, SchedulePlan};
pub use topology::{compute_topological_order, TopoOrder};
pub use transform::rewrite_algebraic;
pub use utils::{format_graph_summary, hash_attributes, sanitize_name, IdGenerator};

/// Framework version string.
pub const VERSION: &str = "0.2.0";