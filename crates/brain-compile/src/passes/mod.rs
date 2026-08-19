//! # Compiler Pass Manager & Pass Pipelines
//!
//! Provides the [`Pass`] abstraction, pass pipeline sequencing, and optimization pipelines.

pub mod broadcast;
pub mod dce;
pub mod fold;
pub mod fusion;
pub mod layout;

use crate::core::{CompilationError, CompileOptions, OptimizationLevel};
use crate::ir::IrGraph;

/// Compiler optimization pass interface.
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, graph: &mut IrGraph) -> Result<bool, CompilationError>;
}

/// Manages and executes a sequence of optimization passes.
#[derive(Default)]
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    /// Creates a new `PassManager`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a pass to the manager.
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Configures standard optimization passes based on compile options.
    pub fn from_options(options: &CompileOptions) -> Self {
        let mut pm = Self::new();

        if options.opt_level >= OptimizationLevel::O1 {
            if options.enable_constant_folding {
                pm.add_pass(Box::new(fold::ConstantFoldingPass));
            }
            if options.enable_dce {
                pm.add_pass(Box::new(dce::DeadCodeEliminationPass));
            }
        }

        if options.opt_level >= OptimizationLevel::O2 && options.enable_fusion {
            pm.add_pass(Box::new(fusion::KernelFusionPass));
        }

        pm
    }

    /// Runs all configured passes sequentially on the IR graph.
    pub fn run(&self, graph: &mut IrGraph) -> Result<bool, CompilationError> {
        let mut changed = false;
        for pass in &self.passes {
            if pass.run(graph)? {
                changed = true;
            }
        }
        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
