//! # Graph Builder API
//!
//! Fluent and incremental API for constructing computational graph IRs.
#![allow(missing_docs)]

use crate::core::{DType, GraphError, GraphResult, Shape, ValueId};
use crate::ir::ops::OpKind;
use crate::ir::GraphIr;

/// Incremental builder for assembling `GraphIr` instances.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    ir: GraphIr,
}

impl GraphBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            ir: GraphIr::new(name),
        }
    }

    /// Adds an input value placeholder to the graph.
    pub fn add_input(&mut self, name: &str, shape: Vec<usize>, dtype: DType) -> ValueId {
        let val_id = self.ir.add_value(name, Shape::new(shape), dtype);
        self.ir.inputs.push(val_id);
        val_id
    }

    /// Adds a constant value to the graph.
    pub fn add_constant(&mut self, name: &str, shape: Vec<usize>, data: Vec<f64>) -> ValueId {
        let val_id = self
            .ir
            .add_value(name, Shape::new(shape.clone()), DType::F32);
        self.ir.set_constant(val_id, data);
        val_id
    }

    /// Adds an operator node consuming `inputs` and producing a newly created output value.
    pub fn add_node(
        &mut self,
        name: &str,
        op: OpKind,
        inputs: Vec<ValueId>,
        output_shape: Vec<usize>,
    ) -> ValueId {
        let out_id = self.ir.add_value(
            &format!("{}_out", name),
            Shape::new(output_shape),
            DType::F32,
        );
        self.ir.add_node(name, op, inputs, vec![out_id]);
        out_id
    }

    /// Marks a value as an output of the graph.
    pub fn mark_output(&mut self, val_id: ValueId) {
        if !self.ir.outputs.contains(&val_id) {
            self.ir.outputs.push(val_id);
        }
    }

    /// Finalizes and returns the built `GraphIr`.
    pub fn build(self) -> GraphResult<GraphIr> {
        if self.ir.nodes.is_empty() && self.ir.inputs.is_empty() {
            return Err(GraphError::VerificationFailed(
                "Cannot build an empty graph".into(),
            ));
        }
        Ok(self.ir)
    }
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
