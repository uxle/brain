//! # Graph Interpreter
//!
//! Pure Rust reference execution runtime interpreting `GraphIr` against `brain_core::Tensor`.
#![allow(missing_docs)]

use std::collections::HashMap;
use brain_core::Tensor;
use crate::core::{GraphResult, GraphError};
use crate::ir::GraphIr;
use crate::ops::op_apply;

/// Execution context maintaining intermediate tensor values.
#[derive(Default)]
pub struct GraphInterpreter {
    values: HashMap<usize, Tensor>,
}

impl GraphInterpreter {
    pub fn new() -> Self {
        Self { values: HashMap::new() }
    }

    pub fn run(&mut self, graph: &GraphIr, inputs: &[Tensor]) -> GraphResult<Vec<Tensor>> {
        self.values.clear();

        if inputs.len() != graph.inputs.len() {
            return Err(GraphError::VerificationFailed(format!(
                "Expected {} inputs, got {}", graph.inputs.len(), inputs.len()
            )));
        }

        // Bind graph inputs
        for (idx, &in_id) in graph.inputs.iter().enumerate() {
            self.values.insert(in_id, inputs[idx].clone());
        }

        // Bind constants
        for (id, val) in graph.values.iter().enumerate() {
            if let Some(ref data) = val.constant_data {
                self.values.insert(id, Tensor::from_vec(data.clone(), val.shape.dims.clone()));
            }
        }

        // Execute nodes sequentially
        for node in &graph.nodes {
            let mut node_inputs = Vec::new();
            for &inp in &node.inputs {
                if let Some(t) = self.values.get(&inp) {
                    node_inputs.push(t);
                } else {
                    return Err(GraphError::ValueNotFound(inp));
                }
            }

            let out_tensor = op_apply(node.op, &node_inputs);
            if let Some(&out_id) = node.outputs.first() {
                self.values.insert(out_id, out_tensor);
            }
        }

        // Gather graph outputs
        let mut outputs = Vec::new();
        for &out_id in &graph.outputs {
            if let Some(t) = self.values.get(&out_id) {
                outputs.push(t.clone());
            } else {
                return Err(GraphError::ValueNotFound(out_id));
            }
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
