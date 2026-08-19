//! # Common Subexpression Elimination (CSE)
//!
//! Detects duplicate operations with identical inputs and merges them.
#![allow(missing_docs)]

use std::collections::HashMap;
use crate::core::GraphResult;
use crate::ir::GraphIr;
use crate::ir::ops::OpKind;
use super::GraphPass;

/// CSE Pass.
#[derive(Debug, Default)]
pub struct CsePass;

impl GraphPass for CsePass {
    fn name(&self) -> &'static str { "CommonSubexpressionElimination" }

    fn run(&mut self, graph: &mut GraphIr) -> GraphResult<bool> {
        eliminate_cse(graph)
    }
}

/// Merges duplicate subexpressions in `GraphIr`.
pub fn eliminate_cse(graph: &mut GraphIr) -> GraphResult<bool> {
    let mut seen_ops: HashMap<(OpKind, Vec<usize>), usize> = HashMap::new();
    let mut value_remap: HashMap<usize, usize> = HashMap::new();
    let mut modified = false;

    for node in &mut graph.nodes {
        // Remap inputs if previously merged
        for inp in &mut node.inputs {
            if let Some(&canonical) = value_remap.get(inp) {
                *inp = canonical;
            }
        }

        let key = (node.op, node.inputs.clone());
        if let Some(&canonical_out) = seen_ops.get(&key) {
            if let Some(&curr_out) = node.outputs.first() {
                value_remap.insert(curr_out, canonical_out);
                modified = true;
            }
        } else if let Some(&curr_out) = node.outputs.first() {
            seen_ops.insert(key, curr_out);
        }
    }

    // Remap graph outputs
    for out in &mut graph.outputs {
        if let Some(&canonical) = value_remap.get(out) {
            *out = canonical;
            modified = true;
        }
    }

    Ok(modified)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
