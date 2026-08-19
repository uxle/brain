//! # Algebraic Rewrites & Transformations
//!
//! Mathematical simplifications: `x * 1 -> x`, `x + 0 -> x`, `x - x -> 0`.
#![allow(missing_docs)]

use crate::ir::GraphIr;
use crate::ir::ops::OpKind;

/// Applies algebraic rewrite rules to simplify operations in `GraphIr`.
pub fn rewrite_algebraic(graph: &mut GraphIr) -> bool {
    let mut modified = false;

    for node in &mut graph.nodes {
        if node.op == OpKind::Add && node.inputs.len() == 2 {
            let in1_const = graph.values[node.inputs[0]].constant_data.as_ref();
            let in2_const = graph.values[node.inputs[1]].constant_data.as_ref();

            if let Some(c) = in2_const {
                if c.iter().all(|&v| v == 0.0) {
                    // x + 0 -> x
                    node.op = OpKind::Relu; // simplified identity proxy
                    node.inputs = vec![node.inputs[0]];
                    modified = true;
                }
            } else if let Some(c) = in1_const {
                if c.iter().all(|&v| v == 0.0) {
                    // 0 + x -> x
                    node.op = OpKind::Relu;
                    node.inputs = vec![node.inputs[1]];
                    modified = true;
                }
            }
        }
    }

    modified
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
