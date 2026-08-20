//! # Optimization Orchestrator
//!
//! Orchestrates passes according to OptLevel and produces an `OptimizeReport`.
#![allow(missing_docs)]

use crate::config::{GraphConfig, OptLevel};
use crate::core::GraphResult;
use crate::ir::GraphIr;
use crate::passes::{
    ConstFoldPass, CsePass, DeadCodeElimPass, FusionPass, InplacePass, PassManager,
};

/// Summary report after graph optimization.
#[derive(Debug, Clone, Default)]
pub struct OptimizeReport {
    pub initial_nodes: usize,
    pub final_nodes: usize,
    pub passes_applied: usize,
}

/// Optimizes a `GraphIr` at the given optimization level.
pub fn optimize(graph: &mut GraphIr, level: OptLevel) -> GraphResult<OptimizeReport> {
    let initial_nodes = graph.nodes.len();
    let config = GraphConfig::for_opt_level(level);
    let mut pm = PassManager::new();

    if config.enable_const_fold {
        pm.add_pass(Box::new(ConstFoldPass));
    }
    if config.enable_cse {
        pm.add_pass(Box::new(CsePass));
    }
    if config.enable_fusion {
        pm.add_pass(Box::new(FusionPass));
    }
    if config.enable_inplace {
        pm.add_pass(Box::new(InplacePass));
    }
    if config.enable_dce {
        pm.add_pass(Box::new(DeadCodeElimPass));
    }

    let iterations = pm.run(graph, config.max_pass_iterations)?;

    Ok(OptimizeReport {
        initial_nodes,
        final_nodes: graph.nodes.len(),
        passes_applied: iterations,
    })
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
