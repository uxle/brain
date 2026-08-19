//! # Brain Compile — Deep Learning JIT Compiler & Optimization Engine
//!
//! Features typed IR lowering, constant folding, dead code elimination, kernel fusion,
//! memory planning, instruction scheduling, and multi-backend code generation (Interpreter, Tensor, Scalar, CUDA, LLVM).
//!
//! ## Subsystems
//!
//! * [`core`] - Compilation targets, optimization levels, and options
//! * [`ir`] - Typed Intermediate Representation (IR), verification, and ops
//! * [`passes`] - Constant folding, DCE/CSE, kernel fusion, broadcast, and layout passes
//! * [`backend`] - Execution backends (Pure-Rust Interpreter, Tensor, Scalar JIT, CUDA, LLVM)
//! * [`jit`] - JIT caching and orchestration engine
//! * [`exec`] - Streaming execution engine
//! * [`plan`] - Liveness memory planner
//! * [`profiler`] - Kernel profiler and roofline analyzer
//! * [`schedule`] - Topological instruction scheduling
//! * [`export_ir`] - Graphviz Dot, JSON, and disassembly exporters
//! * [`transform`] - Algebraic rewrite rules
//! * [`analyze`] - Graph cost model and arithmetic intensity
//! * [`builder`] - Speculative IR builder
//! * [`compute`] - Tensor lifetime analyzer
//! * [`process`] - Multi-stage pipeline runner
//! * [`helper`] - Pattern matching and shape calculation helpers
//!
//! ## Quick Start Example
//!
//! ```rust
//! use brain_compile::prelude::*;
//!
//! let graph = IrGraph::new();
//! let options = CompileOptions::new().with_opt_level(OptimizationLevel::O2);
//! let compiled = compile(&graph, &options).unwrap();
//! assert_eq!(compiled.num_nodes(), 0);
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::derivable_impls,
    clippy::manual_clamp,
    clippy::type_complexity
)]

pub mod analyze;
pub mod backend;
pub mod builder;
pub mod compute;
pub mod config;
pub mod core;
pub mod exec;
pub mod export_ir;
pub mod helper;
pub mod r#impl;
pub mod ir;
pub mod jit;
pub mod ops;
pub mod passes;
pub mod plan;
pub mod process;
pub mod profiler;
pub mod schedule;
pub mod transform;
pub mod utils;

// Re-exports
pub use core::{CompilationError, CompileOptions, OptimizationLevel, TargetBackend};
pub use ir::IrGraph;
pub use r#impl::compile_graph as compile;

/// Package version string.
pub const VERSION: &str = "0.2.0";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 2;
pub const PATCH_VERSION: u32 = 0;

/// Returns the crate version triple.
///
/// ```rust
/// use brain_compile::version_tuple;
/// assert_eq!(version_tuple(), (0, 2, 0));
/// ```
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Returns a formatted version string.
///
/// ```rust
/// use brain_compile::version_string;
/// assert_eq!(version_string(), "brain-compile v0.2.0");
/// ```
pub fn version_string() -> String {
    format!("brain-compile v{}", VERSION)
}

/// Standard prelude imports for compilation utilities.
///
/// ```rust
/// use brain_compile::prelude::*;
/// let opts = CompileOptions::new();
/// assert_eq!(opts.opt_level, OptimizationLevel::O2);
/// ```
pub mod prelude {
    pub use crate::core::{CompilationError, CompileOptions, OptimizationLevel, TargetBackend};
    pub use crate::ir::{IrGraph, IrOp, IrType, IrValue};
    pub use crate::r#impl::compile_graph as compile;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
