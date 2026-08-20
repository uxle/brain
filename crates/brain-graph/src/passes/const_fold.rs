//! # Constant Folding Optimization Pass
//!
//! Evaluates subexpressions involving purely constant inputs at compile time.
#![allow(missing_docs)]

use super::GraphPass;
use crate::core::GraphResult;
use crate::ir::ops::OpKind;
use crate::ir::GraphIr;
use crate::ops::op_apply;
use brain_core::Tensor;

/// Constant folding pass implementation.
#[derive(Debug, Default)]
pub struct ConstFoldPass;

impl GraphPass for ConstFoldPass {
    fn name(&self) -> &'static str {
        "ConstantFolding"
    }

    fn run(&mut self, graph: &mut GraphIr) -> GraphResult<bool> {
        fold_constants(graph)
    }
}

/// Folds constant operations in `GraphIr`. Returns true if any node was folded.
pub fn fold_constants(graph: &mut GraphIr) -> GraphResult<bool> {
    let mut modified = false;

    for node in &mut graph.nodes {
        // Check if all inputs are constant
        if node.inputs.is_empty() {
            continue;
        }
        let all_const = node
            .inputs
            .iter()
            .all(|&inp| graph.values[inp].constant_data.is_some());

        if all_const && node.outputs.len() == 1 {
            let input_tensors: Vec<Tensor> = node
                .inputs
                .iter()
                .map(|&inp| {
                    let v = &graph.values[inp];
                    Tensor::from_vec(
                        v.constant_data.as_ref().unwrap().clone(),
                        v.shape.dims.clone(),
                    )
                })
                .collect();

            let refs: Vec<&Tensor> = input_tensors.iter().collect();
            let res = op_apply(node.op, &refs);

            let out_val = node.outputs[0];
            graph.values[out_val].constant_data = Some(res.to_vec());
            node.op = OpKind::Constant;
            node.inputs.clear();
            modified = true;
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
