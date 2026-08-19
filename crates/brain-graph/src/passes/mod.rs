//! # Graph Optimization Passes
//!
//! Pass infrastructure, [`GraphPass`] trait, PassManager, and pass pipeline coordinator.
#![allow(missing_docs)]

pub mod const_fold;
pub mod dead_code;
pub mod cse;
pub mod fusion;
pub mod layout;
pub mod inplace;

pub use const_fold::{fold_constants, ConstFoldPass};
pub use dead_code::{eliminate_dead_code, DeadCodeElimPass};
pub use cse::{eliminate_cse, CsePass};
pub use fusion::{plan_fusion, FusionPass, FusionPlan};
pub use layout::{eliminate_layout_transforms, LayoutPass};
pub use inplace::{plan_inplace_operations, InplacePass, InplacePlan};

use crate::core::GraphResult;
use crate::ir::GraphIr;

/// Core trait implemented by all optimization passes.
pub trait GraphPass {
    /// Name of the optimization pass.
    fn name(&self) -> &'static str;
    /// Executes the pass on the target `GraphIr`. Returns true if graph was modified.
    fn run(&mut self, graph: &mut GraphIr) -> GraphResult<bool>;
}

/// Orchestrates execution of a sequence of optimization passes.
#[derive(Default)]
pub struct PassManager {
    passes: Vec<Box<dyn GraphPass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(&mut self, pass: Box<dyn GraphPass>) {
        self.passes.push(pass);
    }

    pub fn run(&mut self, graph: &mut GraphIr, max_iterations: usize) -> GraphResult<usize> {
        let mut total_iterations = 0;
        for iter in 0..max_iterations {
            total_iterations = iter + 1;
            let mut modified = false;
            for pass in self.passes.iter_mut() {
                if pass.run(graph)? {
                    modified = true;
                }
            }
            if !modified {
                break;
            }
        }
        Ok(total_iterations)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
