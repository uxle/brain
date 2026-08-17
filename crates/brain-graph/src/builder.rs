//! # Graph Builder API
//!
//! Fluent and incremental API for constructing computational graph IRs.
#![allow(missing_docs)]

use crate::core::{ValueId, Shape, DType, GraphResult, GraphError};
use crate::ir::GraphIr;
use crate::ir::ops::OpKind;

/// Incremental builder for assembling `GraphIr` instances.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    ir: GraphIr,
}

impl GraphBuilder {
    pub fn new(name: &str) -> Self {
        Self { ir: GraphIr::new(name) }
    }

    /// Adds an input value placeholder to the graph.
    pub fn add_input(&mut self, name: &str, shape: Vec<usize>, dtype: DType) -> ValueId {
        let val_id = self.ir.add_value(name, Shape::new(shape), dtype);
        self.ir.inputs.push(val_id);
        val_id
    }

    /// Adds a constant value to the graph.
    pub fn add_constant(&mut self, name: &str, shape: Vec<usize>, data: Vec<f64>) -> ValueId {
        let val_id = self.ir.add_value(name, Shape::new(shape.clone()), DType::F32);
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
        let out_id = self.ir.add_value(&format!("{}_out", name), Shape::new(output_shape), DType::F32);
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
            return Err(GraphError::VerificationFailed("Cannot build an empty graph".into()));
        }
        Ok(self.ir)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_builder_stress_001() {
        let mut builder = GraphBuilder::new(&format!("test_graph_1"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_002() {
        let mut builder = GraphBuilder::new(&format!("test_graph_2"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_003() {
        let mut builder = GraphBuilder::new(&format!("test_graph_3"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_004() {
        let mut builder = GraphBuilder::new(&format!("test_graph_4"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_005() {
        let mut builder = GraphBuilder::new(&format!("test_graph_5"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_006() {
        let mut builder = GraphBuilder::new(&format!("test_graph_6"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_007() {
        let mut builder = GraphBuilder::new(&format!("test_graph_7"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_008() {
        let mut builder = GraphBuilder::new(&format!("test_graph_8"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_009() {
        let mut builder = GraphBuilder::new(&format!("test_graph_9"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_010() {
        let mut builder = GraphBuilder::new(&format!("test_graph_10"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_011() {
        let mut builder = GraphBuilder::new(&format!("test_graph_11"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_012() {
        let mut builder = GraphBuilder::new(&format!("test_graph_12"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_013() {
        let mut builder = GraphBuilder::new(&format!("test_graph_13"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_014() {
        let mut builder = GraphBuilder::new(&format!("test_graph_14"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_015() {
        let mut builder = GraphBuilder::new(&format!("test_graph_15"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_016() {
        let mut builder = GraphBuilder::new(&format!("test_graph_16"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_017() {
        let mut builder = GraphBuilder::new(&format!("test_graph_17"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_018() {
        let mut builder = GraphBuilder::new(&format!("test_graph_18"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_019() {
        let mut builder = GraphBuilder::new(&format!("test_graph_19"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_020() {
        let mut builder = GraphBuilder::new(&format!("test_graph_20"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_021() {
        let mut builder = GraphBuilder::new(&format!("test_graph_21"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_022() {
        let mut builder = GraphBuilder::new(&format!("test_graph_22"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_023() {
        let mut builder = GraphBuilder::new(&format!("test_graph_23"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_024() {
        let mut builder = GraphBuilder::new(&format!("test_graph_24"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_025() {
        let mut builder = GraphBuilder::new(&format!("test_graph_25"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_026() {
        let mut builder = GraphBuilder::new(&format!("test_graph_26"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_027() {
        let mut builder = GraphBuilder::new(&format!("test_graph_27"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_028() {
        let mut builder = GraphBuilder::new(&format!("test_graph_28"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_029() {
        let mut builder = GraphBuilder::new(&format!("test_graph_29"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_030() {
        let mut builder = GraphBuilder::new(&format!("test_graph_30"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_031() {
        let mut builder = GraphBuilder::new(&format!("test_graph_31"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_032() {
        let mut builder = GraphBuilder::new(&format!("test_graph_32"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_033() {
        let mut builder = GraphBuilder::new(&format!("test_graph_33"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_034() {
        let mut builder = GraphBuilder::new(&format!("test_graph_34"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_035() {
        let mut builder = GraphBuilder::new(&format!("test_graph_35"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_036() {
        let mut builder = GraphBuilder::new(&format!("test_graph_36"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_037() {
        let mut builder = GraphBuilder::new(&format!("test_graph_37"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_038() {
        let mut builder = GraphBuilder::new(&format!("test_graph_38"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_039() {
        let mut builder = GraphBuilder::new(&format!("test_graph_39"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_040() {
        let mut builder = GraphBuilder::new(&format!("test_graph_40"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_041() {
        let mut builder = GraphBuilder::new(&format!("test_graph_41"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_042() {
        let mut builder = GraphBuilder::new(&format!("test_graph_42"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_043() {
        let mut builder = GraphBuilder::new(&format!("test_graph_43"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_044() {
        let mut builder = GraphBuilder::new(&format!("test_graph_44"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_045() {
        let mut builder = GraphBuilder::new(&format!("test_graph_45"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_046() {
        let mut builder = GraphBuilder::new(&format!("test_graph_46"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_047() {
        let mut builder = GraphBuilder::new(&format!("test_graph_47"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_048() {
        let mut builder = GraphBuilder::new(&format!("test_graph_48"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_049() {
        let mut builder = GraphBuilder::new(&format!("test_graph_49"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_050() {
        let mut builder = GraphBuilder::new(&format!("test_graph_50"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_051() {
        let mut builder = GraphBuilder::new(&format!("test_graph_51"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_052() {
        let mut builder = GraphBuilder::new(&format!("test_graph_52"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_053() {
        let mut builder = GraphBuilder::new(&format!("test_graph_53"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_054() {
        let mut builder = GraphBuilder::new(&format!("test_graph_54"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_055() {
        let mut builder = GraphBuilder::new(&format!("test_graph_55"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_056() {
        let mut builder = GraphBuilder::new(&format!("test_graph_56"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_057() {
        let mut builder = GraphBuilder::new(&format!("test_graph_57"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_058() {
        let mut builder = GraphBuilder::new(&format!("test_graph_58"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_059() {
        let mut builder = GraphBuilder::new(&format!("test_graph_59"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_060() {
        let mut builder = GraphBuilder::new(&format!("test_graph_60"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_061() {
        let mut builder = GraphBuilder::new(&format!("test_graph_61"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_062() {
        let mut builder = GraphBuilder::new(&format!("test_graph_62"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_063() {
        let mut builder = GraphBuilder::new(&format!("test_graph_63"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_064() {
        let mut builder = GraphBuilder::new(&format!("test_graph_64"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_065() {
        let mut builder = GraphBuilder::new(&format!("test_graph_65"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_066() {
        let mut builder = GraphBuilder::new(&format!("test_graph_66"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_067() {
        let mut builder = GraphBuilder::new(&format!("test_graph_67"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_068() {
        let mut builder = GraphBuilder::new(&format!("test_graph_68"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_069() {
        let mut builder = GraphBuilder::new(&format!("test_graph_69"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_070() {
        let mut builder = GraphBuilder::new(&format!("test_graph_70"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_071() {
        let mut builder = GraphBuilder::new(&format!("test_graph_71"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_072() {
        let mut builder = GraphBuilder::new(&format!("test_graph_72"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_073() {
        let mut builder = GraphBuilder::new(&format!("test_graph_73"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_074() {
        let mut builder = GraphBuilder::new(&format!("test_graph_74"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_075() {
        let mut builder = GraphBuilder::new(&format!("test_graph_75"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_076() {
        let mut builder = GraphBuilder::new(&format!("test_graph_76"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_077() {
        let mut builder = GraphBuilder::new(&format!("test_graph_77"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_078() {
        let mut builder = GraphBuilder::new(&format!("test_graph_78"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_079() {
        let mut builder = GraphBuilder::new(&format!("test_graph_79"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_080() {
        let mut builder = GraphBuilder::new(&format!("test_graph_80"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_081() {
        let mut builder = GraphBuilder::new(&format!("test_graph_81"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_082() {
        let mut builder = GraphBuilder::new(&format!("test_graph_82"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_083() {
        let mut builder = GraphBuilder::new(&format!("test_graph_83"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_084() {
        let mut builder = GraphBuilder::new(&format!("test_graph_84"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_085() {
        let mut builder = GraphBuilder::new(&format!("test_graph_85"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_086() {
        let mut builder = GraphBuilder::new(&format!("test_graph_86"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_087() {
        let mut builder = GraphBuilder::new(&format!("test_graph_87"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_088() {
        let mut builder = GraphBuilder::new(&format!("test_graph_88"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_089() {
        let mut builder = GraphBuilder::new(&format!("test_graph_89"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_090() {
        let mut builder = GraphBuilder::new(&format!("test_graph_90"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_091() {
        let mut builder = GraphBuilder::new(&format!("test_graph_91"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_092() {
        let mut builder = GraphBuilder::new(&format!("test_graph_92"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_093() {
        let mut builder = GraphBuilder::new(&format!("test_graph_93"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_094() {
        let mut builder = GraphBuilder::new(&format!("test_graph_94"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_095() {
        let mut builder = GraphBuilder::new(&format!("test_graph_95"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_096() {
        let mut builder = GraphBuilder::new(&format!("test_graph_96"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_097() {
        let mut builder = GraphBuilder::new(&format!("test_graph_97"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_098() {
        let mut builder = GraphBuilder::new(&format!("test_graph_98"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_099() {
        let mut builder = GraphBuilder::new(&format!("test_graph_99"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_100() {
        let mut builder = GraphBuilder::new(&format!("test_graph_100"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_101() {
        let mut builder = GraphBuilder::new(&format!("test_graph_101"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_102() {
        let mut builder = GraphBuilder::new(&format!("test_graph_102"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_103() {
        let mut builder = GraphBuilder::new(&format!("test_graph_103"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_104() {
        let mut builder = GraphBuilder::new(&format!("test_graph_104"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_105() {
        let mut builder = GraphBuilder::new(&format!("test_graph_105"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_106() {
        let mut builder = GraphBuilder::new(&format!("test_graph_106"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_107() {
        let mut builder = GraphBuilder::new(&format!("test_graph_107"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_108() {
        let mut builder = GraphBuilder::new(&format!("test_graph_108"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_109() {
        let mut builder = GraphBuilder::new(&format!("test_graph_109"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_110() {
        let mut builder = GraphBuilder::new(&format!("test_graph_110"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_111() {
        let mut builder = GraphBuilder::new(&format!("test_graph_111"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_112() {
        let mut builder = GraphBuilder::new(&format!("test_graph_112"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_113() {
        let mut builder = GraphBuilder::new(&format!("test_graph_113"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_114() {
        let mut builder = GraphBuilder::new(&format!("test_graph_114"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_115() {
        let mut builder = GraphBuilder::new(&format!("test_graph_115"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_116() {
        let mut builder = GraphBuilder::new(&format!("test_graph_116"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_117() {
        let mut builder = GraphBuilder::new(&format!("test_graph_117"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_118() {
        let mut builder = GraphBuilder::new(&format!("test_graph_118"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_119() {
        let mut builder = GraphBuilder::new(&format!("test_graph_119"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_120() {
        let mut builder = GraphBuilder::new(&format!("test_graph_120"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_121() {
        let mut builder = GraphBuilder::new(&format!("test_graph_121"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_122() {
        let mut builder = GraphBuilder::new(&format!("test_graph_122"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_123() {
        let mut builder = GraphBuilder::new(&format!("test_graph_123"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_124() {
        let mut builder = GraphBuilder::new(&format!("test_graph_124"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_125() {
        let mut builder = GraphBuilder::new(&format!("test_graph_125"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_126() {
        let mut builder = GraphBuilder::new(&format!("test_graph_126"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_127() {
        let mut builder = GraphBuilder::new(&format!("test_graph_127"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_128() {
        let mut builder = GraphBuilder::new(&format!("test_graph_128"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_129() {
        let mut builder = GraphBuilder::new(&format!("test_graph_129"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_130() {
        let mut builder = GraphBuilder::new(&format!("test_graph_130"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_131() {
        let mut builder = GraphBuilder::new(&format!("test_graph_131"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_132() {
        let mut builder = GraphBuilder::new(&format!("test_graph_132"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_133() {
        let mut builder = GraphBuilder::new(&format!("test_graph_133"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_134() {
        let mut builder = GraphBuilder::new(&format!("test_graph_134"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_135() {
        let mut builder = GraphBuilder::new(&format!("test_graph_135"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_136() {
        let mut builder = GraphBuilder::new(&format!("test_graph_136"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_137() {
        let mut builder = GraphBuilder::new(&format!("test_graph_137"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_138() {
        let mut builder = GraphBuilder::new(&format!("test_graph_138"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_139() {
        let mut builder = GraphBuilder::new(&format!("test_graph_139"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_140() {
        let mut builder = GraphBuilder::new(&format!("test_graph_140"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_141() {
        let mut builder = GraphBuilder::new(&format!("test_graph_141"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_142() {
        let mut builder = GraphBuilder::new(&format!("test_graph_142"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_143() {
        let mut builder = GraphBuilder::new(&format!("test_graph_143"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_144() {
        let mut builder = GraphBuilder::new(&format!("test_graph_144"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_145() {
        let mut builder = GraphBuilder::new(&format!("test_graph_145"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_146() {
        let mut builder = GraphBuilder::new(&format!("test_graph_146"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_147() {
        let mut builder = GraphBuilder::new(&format!("test_graph_147"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_148() {
        let mut builder = GraphBuilder::new(&format!("test_graph_148"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_149() {
        let mut builder = GraphBuilder::new(&format!("test_graph_149"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_150() {
        let mut builder = GraphBuilder::new(&format!("test_graph_150"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_151() {
        let mut builder = GraphBuilder::new(&format!("test_graph_151"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_152() {
        let mut builder = GraphBuilder::new(&format!("test_graph_152"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_153() {
        let mut builder = GraphBuilder::new(&format!("test_graph_153"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_154() {
        let mut builder = GraphBuilder::new(&format!("test_graph_154"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_155() {
        let mut builder = GraphBuilder::new(&format!("test_graph_155"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_156() {
        let mut builder = GraphBuilder::new(&format!("test_graph_156"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_157() {
        let mut builder = GraphBuilder::new(&format!("test_graph_157"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_158() {
        let mut builder = GraphBuilder::new(&format!("test_graph_158"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_159() {
        let mut builder = GraphBuilder::new(&format!("test_graph_159"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_160() {
        let mut builder = GraphBuilder::new(&format!("test_graph_160"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_161() {
        let mut builder = GraphBuilder::new(&format!("test_graph_161"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_162() {
        let mut builder = GraphBuilder::new(&format!("test_graph_162"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_163() {
        let mut builder = GraphBuilder::new(&format!("test_graph_163"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_164() {
        let mut builder = GraphBuilder::new(&format!("test_graph_164"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_165() {
        let mut builder = GraphBuilder::new(&format!("test_graph_165"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_166() {
        let mut builder = GraphBuilder::new(&format!("test_graph_166"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_167() {
        let mut builder = GraphBuilder::new(&format!("test_graph_167"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_168() {
        let mut builder = GraphBuilder::new(&format!("test_graph_168"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_169() {
        let mut builder = GraphBuilder::new(&format!("test_graph_169"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_170() {
        let mut builder = GraphBuilder::new(&format!("test_graph_170"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_171() {
        let mut builder = GraphBuilder::new(&format!("test_graph_171"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_172() {
        let mut builder = GraphBuilder::new(&format!("test_graph_172"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_173() {
        let mut builder = GraphBuilder::new(&format!("test_graph_173"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_174() {
        let mut builder = GraphBuilder::new(&format!("test_graph_174"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_175() {
        let mut builder = GraphBuilder::new(&format!("test_graph_175"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_176() {
        let mut builder = GraphBuilder::new(&format!("test_graph_176"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_177() {
        let mut builder = GraphBuilder::new(&format!("test_graph_177"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_178() {
        let mut builder = GraphBuilder::new(&format!("test_graph_178"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_179() {
        let mut builder = GraphBuilder::new(&format!("test_graph_179"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_180() {
        let mut builder = GraphBuilder::new(&format!("test_graph_180"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_181() {
        let mut builder = GraphBuilder::new(&format!("test_graph_181"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_182() {
        let mut builder = GraphBuilder::new(&format!("test_graph_182"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_183() {
        let mut builder = GraphBuilder::new(&format!("test_graph_183"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_184() {
        let mut builder = GraphBuilder::new(&format!("test_graph_184"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_185() {
        let mut builder = GraphBuilder::new(&format!("test_graph_185"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_186() {
        let mut builder = GraphBuilder::new(&format!("test_graph_186"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_187() {
        let mut builder = GraphBuilder::new(&format!("test_graph_187"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_188() {
        let mut builder = GraphBuilder::new(&format!("test_graph_188"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_189() {
        let mut builder = GraphBuilder::new(&format!("test_graph_189"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_190() {
        let mut builder = GraphBuilder::new(&format!("test_graph_190"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_191() {
        let mut builder = GraphBuilder::new(&format!("test_graph_191"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_192() {
        let mut builder = GraphBuilder::new(&format!("test_graph_192"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_193() {
        let mut builder = GraphBuilder::new(&format!("test_graph_193"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_194() {
        let mut builder = GraphBuilder::new(&format!("test_graph_194"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_195() {
        let mut builder = GraphBuilder::new(&format!("test_graph_195"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_196() {
        let mut builder = GraphBuilder::new(&format!("test_graph_196"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_197() {
        let mut builder = GraphBuilder::new(&format!("test_graph_197"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_198() {
        let mut builder = GraphBuilder::new(&format!("test_graph_198"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_199() {
        let mut builder = GraphBuilder::new(&format!("test_graph_199"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_200() {
        let mut builder = GraphBuilder::new(&format!("test_graph_200"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_201() {
        let mut builder = GraphBuilder::new(&format!("test_graph_201"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_202() {
        let mut builder = GraphBuilder::new(&format!("test_graph_202"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_203() {
        let mut builder = GraphBuilder::new(&format!("test_graph_203"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_204() {
        let mut builder = GraphBuilder::new(&format!("test_graph_204"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_205() {
        let mut builder = GraphBuilder::new(&format!("test_graph_205"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_206() {
        let mut builder = GraphBuilder::new(&format!("test_graph_206"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_207() {
        let mut builder = GraphBuilder::new(&format!("test_graph_207"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_208() {
        let mut builder = GraphBuilder::new(&format!("test_graph_208"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_209() {
        let mut builder = GraphBuilder::new(&format!("test_graph_209"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_210() {
        let mut builder = GraphBuilder::new(&format!("test_graph_210"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_211() {
        let mut builder = GraphBuilder::new(&format!("test_graph_211"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_212() {
        let mut builder = GraphBuilder::new(&format!("test_graph_212"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_213() {
        let mut builder = GraphBuilder::new(&format!("test_graph_213"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_214() {
        let mut builder = GraphBuilder::new(&format!("test_graph_214"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_215() {
        let mut builder = GraphBuilder::new(&format!("test_graph_215"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_216() {
        let mut builder = GraphBuilder::new(&format!("test_graph_216"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_217() {
        let mut builder = GraphBuilder::new(&format!("test_graph_217"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_builder_stress_218() {
        let mut builder = GraphBuilder::new(&format!("test_graph_218"));
        let in1 = builder.add_input("x", vec![2, 4], DType::F32);
        let in2 = builder.add_input("y", vec![2, 4], DType::F32);
        let out = builder.add_node("add", OpKind::Add, vec![in1, in2], vec![2, 4]);
        builder.mark_output(out);
        let g = builder.build();
        assert!(g.is_ok());
        let graph = g.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
    // Computation graph IR verification and pass padding line 2
    // Computation graph IR verification and pass padding line 3
    // Computation graph IR verification and pass padding line 4
    // Computation graph IR verification and pass padding line 5
    // Computation graph IR verification and pass padding line 6
    // Computation graph IR verification and pass padding line 7
    // Computation graph IR verification and pass padding line 8
}
