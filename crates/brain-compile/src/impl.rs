//! # Main Compilation Engine Coordinator
//!
//! Orchestrates verification, optimization passes, scheduling, and backend emission.

use crate::core::{CompilationError, CompileOptions};
use crate::ir::IrGraph;
use crate::passes::PassManager;

/// Compiles and optimizes an IR graph according to the given compilation options.
pub fn compile_graph(graph: &IrGraph, options: &CompileOptions) -> Result<IrGraph, CompilationError> {
    // 1. Verify initial IR
    crate::ir::verify::verify_graph(graph)?;

    // 2. Run optimization passes
    let mut optimized = graph.clone();
    let pm = PassManager::from_options(options);
    pm.run(&mut optimized)?;

    // 3. Verify optimized IR
    crate::ir::verify::verify_graph(&optimized)?;

    Ok(optimized)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
