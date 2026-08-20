//! # Graph Shape Inference
//!
//! Propagates tensor dimensions across graph operators forward from inputs.
#![allow(missing_docs)]

use crate::core::{GraphResult, Shape};
use crate::ir::ops::OpKind;
use crate::ir::GraphIr;

/// Carries status and shape inferences across the graph.
#[derive(Debug, Clone, Default)]
pub struct ShapeInferenceResult {
    pub inferred_shapes: Vec<Shape>,
}

/// Infers and updates all output shapes in a `GraphIr`.
pub fn infer_graph_shapes(graph: &mut GraphIr) -> GraphResult<ShapeInferenceResult> {
    let mut inferred = Vec::with_capacity(graph.values.len());

    for v in &graph.values {
        inferred.push(v.shape.clone());
    }

    for node in &graph.nodes {
        match node.op {
            OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Relu
            | OpKind::Sigmoid
            | OpKind::Tanh
            | OpKind::Gelu => {
                if let Some(&first_in) = node.inputs.first() {
                    let in_shape = inferred[first_in].clone();
                    for &out in &node.outputs {
                        inferred[out] = in_shape.clone();
                        graph.values[out].shape = in_shape.clone();
                    }
                }
            }
            OpKind::MatMul => {
                if node.inputs.len() >= 2 {
                    let s_a = &inferred[node.inputs[0]].dims;
                    let s_b = &inferred[node.inputs[1]].dims;
                    if s_a.len() == 2 && s_b.len() == 2 && s_a[1] == s_b[0] {
                        let out_shape = Shape::new(vec![s_a[0], s_b[1]]);
                        for &out in &node.outputs {
                            inferred[out] = out_shape.clone();
                            graph.values[out].shape = out_shape.clone();
                        }
                    }
                }
            }
            OpKind::Flatten => {
                if let Some(&first_in) = node.inputs.first() {
                    let total: usize = inferred[first_in].dims.iter().product();
                    let out_shape = Shape::new(vec![total]);
                    for &out in &node.outputs {
                        inferred[out] = out_shape.clone();
                        graph.values[out].shape = out_shape.clone();
                    }
                }
            }
            _ => {}
        }
    }

    Ok(ShapeInferenceResult {
        inferred_shapes: inferred,
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
