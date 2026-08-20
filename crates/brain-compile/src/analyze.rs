//! # Graph Cost Model & FLOP Analysis
//!
//! Evaluates computational complexity, FLOP estimates, and memory bandwidth requirements.

use crate::ir::{IrGraph, OpKind};

/// Computes the total theoretical floating-point operations in an IR graph.
pub fn estimate_total_flops(graph: &IrGraph) -> u64 {
    let mut total_flops = 0u64;

    for node in &graph.nodes {
        let out_val = &graph.values[node.output];
        let numel = out_val.numel() as u64;

        match &node.kind {
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
                total_flops += numel;
            }
            OpKind::MatMul => {
                let k = if node.inputs.len() >= 2 {
                    let in0_shape = &graph.values[node.inputs[0]].shape;
                    if in0_shape.len() >= 2 {
                        in0_shape[1] as u64
                    } else {
                        1
                    }
                } else {
                    1
                };
                total_flops += 2 * numel * k;
            }
            OpKind::Exp | OpKind::Log | OpKind::Sin | OpKind::Cos | OpKind::Tanh => {
                total_flops += 4 * numel;
            }
            _ => {
                total_flops += numel;
            }
        }
    }

    total_flops
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
