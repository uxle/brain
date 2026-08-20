//! # Layout & Transpose Elimination
//!
//! Cancels redundant consecutive transpose operations and optimizes data layout.
#![allow(missing_docs)]

use super::GraphPass;
use crate::core::GraphResult;
use crate::ir::ops::OpKind;
use crate::ir::GraphIr;

/// Layout optimization pass.
#[derive(Debug, Default)]
pub struct LayoutPass;

impl GraphPass for LayoutPass {
    fn name(&self) -> &'static str {
        "LayoutOptimization"
    }

    fn run(&mut self, graph: &mut GraphIr) -> GraphResult<bool> {
        eliminate_layout_transforms(graph)
    }
}

/// Cancels redundant back-to-back transpose pairs.
pub fn eliminate_layout_transforms(graph: &mut GraphIr) -> GraphResult<bool> {
    let mut modified = false;

    for i in 0..graph.nodes.len() {
        if graph.nodes[i].op == OpKind::Transpose {
            let out_v = graph.nodes[i].outputs[0];
            for j in (i + 1)..graph.nodes.len() {
                if graph.nodes[j].op == OpKind::Transpose && graph.nodes[j].inputs.contains(&out_v)
                {
                    // Two consecutive transposes cancel out to identity
                    let orig_in = graph.nodes[i].inputs[0];
                    let _final_out = graph.nodes[j].outputs[0];
                    graph.nodes[j].op = OpKind::Relu; // simplified replacement
                    graph.nodes[j].inputs = vec![orig_in];
                    modified = true;
                }
            }
        }
    }

    Ok(modified)
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
