//! # Graph Intermediate Representation (IR)
//!
//! Representation of computation nodes, dataflow edges, and intermediate values.
#![allow(missing_docs)]

pub mod ops;
pub mod shape_infer;
pub mod verify;

pub use ops::{OpKind, OpRegistry};
pub use shape_infer::infer_graph_shapes;
pub use verify::verify_graph;

use crate::core::{DType, EdgeId, NodeId, Shape, ValueId};
use std::collections::HashMap;

/// Node representing an operation in the dataflow computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub name: String,
    pub op: OpKind,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub attributes: HashMap<String, String>,
}

/// Value representing a tensor flowing between operators.
#[derive(Debug, Clone)]
pub struct GraphValue {
    pub id: ValueId,
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
    pub constant_data: Option<Vec<f64>>,
}

/// Explicit edge connecting producer node to consumer node.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub id: EdgeId,
    pub src_node: NodeId,
    pub dst_node: NodeId,
    pub value_id: ValueId,
}

/// Full computation graph IR.
#[derive(Debug, Clone, Default)]
pub struct GraphIr {
    pub name: String,
    pub nodes: Vec<GraphNode>,
    pub values: Vec<GraphValue>,
    pub edges: Vec<GraphEdge>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

impl GraphIr {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: Vec::new(),
            values: Vec::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn add_value(&mut self, name: &str, shape: Shape, dtype: DType) -> ValueId {
        let id = self.values.len();
        self.values.push(GraphValue {
            id,
            name: name.to_string(),
            shape,
            dtype,
            constant_data: None,
        });
        id
    }

    pub fn set_constant(&mut self, id: ValueId, data: Vec<f64>) {
        if let Some(v) = self.values.get_mut(id) {
            v.constant_data = Some(data);
        }
    }

    pub fn add_node(
        &mut self,
        name: &str,
        op: OpKind,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    ) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(GraphNode {
            id,
            name: name.to_string(),
            op,
            inputs,
            outputs,
            attributes: HashMap::new(),
        });
        id
    }

    pub fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    pub fn get_value(&self, id: ValueId) -> Option<&GraphValue> {
        self.values.get(id)
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_values(&self) -> usize {
        self.values.len()
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
}
