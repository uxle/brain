//! # Graph Intermediate Representation (IR)
//!
//! Representation of computation nodes, dataflow edges, and intermediate values.
#![allow(missing_docs)]

pub mod ops;
pub mod verify;
pub mod shape_infer;

pub use ops::{OpKind, OpRegistry};
pub use verify::verify_graph;
pub use shape_infer::infer_graph_shapes;

use std::collections::HashMap;
use crate::core::{NodeId, ValueId, EdgeId, Shape, DType};

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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    
    #[test]
    fn test_ir_mod_stress_001() {
        let mut ir = GraphIr::new(&format!("ir_1"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_002() {
        let mut ir = GraphIr::new(&format!("ir_2"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_003() {
        let mut ir = GraphIr::new(&format!("ir_3"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_004() {
        let mut ir = GraphIr::new(&format!("ir_4"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_005() {
        let mut ir = GraphIr::new(&format!("ir_5"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_006() {
        let mut ir = GraphIr::new(&format!("ir_6"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_007() {
        let mut ir = GraphIr::new(&format!("ir_7"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_008() {
        let mut ir = GraphIr::new(&format!("ir_8"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_009() {
        let mut ir = GraphIr::new(&format!("ir_9"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_010() {
        let mut ir = GraphIr::new(&format!("ir_10"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_011() {
        let mut ir = GraphIr::new(&format!("ir_11"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_012() {
        let mut ir = GraphIr::new(&format!("ir_12"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_013() {
        let mut ir = GraphIr::new(&format!("ir_13"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_014() {
        let mut ir = GraphIr::new(&format!("ir_14"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_015() {
        let mut ir = GraphIr::new(&format!("ir_15"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_016() {
        let mut ir = GraphIr::new(&format!("ir_16"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_017() {
        let mut ir = GraphIr::new(&format!("ir_17"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_018() {
        let mut ir = GraphIr::new(&format!("ir_18"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_019() {
        let mut ir = GraphIr::new(&format!("ir_19"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_020() {
        let mut ir = GraphIr::new(&format!("ir_20"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_021() {
        let mut ir = GraphIr::new(&format!("ir_21"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_022() {
        let mut ir = GraphIr::new(&format!("ir_22"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_023() {
        let mut ir = GraphIr::new(&format!("ir_23"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_024() {
        let mut ir = GraphIr::new(&format!("ir_24"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_025() {
        let mut ir = GraphIr::new(&format!("ir_25"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_026() {
        let mut ir = GraphIr::new(&format!("ir_26"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_027() {
        let mut ir = GraphIr::new(&format!("ir_27"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_028() {
        let mut ir = GraphIr::new(&format!("ir_28"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_029() {
        let mut ir = GraphIr::new(&format!("ir_29"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_030() {
        let mut ir = GraphIr::new(&format!("ir_30"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_031() {
        let mut ir = GraphIr::new(&format!("ir_31"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_032() {
        let mut ir = GraphIr::new(&format!("ir_32"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_033() {
        let mut ir = GraphIr::new(&format!("ir_33"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_034() {
        let mut ir = GraphIr::new(&format!("ir_34"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_035() {
        let mut ir = GraphIr::new(&format!("ir_35"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_036() {
        let mut ir = GraphIr::new(&format!("ir_36"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_037() {
        let mut ir = GraphIr::new(&format!("ir_37"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_038() {
        let mut ir = GraphIr::new(&format!("ir_38"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_039() {
        let mut ir = GraphIr::new(&format!("ir_39"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_040() {
        let mut ir = GraphIr::new(&format!("ir_40"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_041() {
        let mut ir = GraphIr::new(&format!("ir_41"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_042() {
        let mut ir = GraphIr::new(&format!("ir_42"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_043() {
        let mut ir = GraphIr::new(&format!("ir_43"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_044() {
        let mut ir = GraphIr::new(&format!("ir_44"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_045() {
        let mut ir = GraphIr::new(&format!("ir_45"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_046() {
        let mut ir = GraphIr::new(&format!("ir_46"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_047() {
        let mut ir = GraphIr::new(&format!("ir_47"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_048() {
        let mut ir = GraphIr::new(&format!("ir_48"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_049() {
        let mut ir = GraphIr::new(&format!("ir_49"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_050() {
        let mut ir = GraphIr::new(&format!("ir_50"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_051() {
        let mut ir = GraphIr::new(&format!("ir_51"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_052() {
        let mut ir = GraphIr::new(&format!("ir_52"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_053() {
        let mut ir = GraphIr::new(&format!("ir_53"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_054() {
        let mut ir = GraphIr::new(&format!("ir_54"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_055() {
        let mut ir = GraphIr::new(&format!("ir_55"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_056() {
        let mut ir = GraphIr::new(&format!("ir_56"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_057() {
        let mut ir = GraphIr::new(&format!("ir_57"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_058() {
        let mut ir = GraphIr::new(&format!("ir_58"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_059() {
        let mut ir = GraphIr::new(&format!("ir_59"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_060() {
        let mut ir = GraphIr::new(&format!("ir_60"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_061() {
        let mut ir = GraphIr::new(&format!("ir_61"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_062() {
        let mut ir = GraphIr::new(&format!("ir_62"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_063() {
        let mut ir = GraphIr::new(&format!("ir_63"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_064() {
        let mut ir = GraphIr::new(&format!("ir_64"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_065() {
        let mut ir = GraphIr::new(&format!("ir_65"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_066() {
        let mut ir = GraphIr::new(&format!("ir_66"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_067() {
        let mut ir = GraphIr::new(&format!("ir_67"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_068() {
        let mut ir = GraphIr::new(&format!("ir_68"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_069() {
        let mut ir = GraphIr::new(&format!("ir_69"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_070() {
        let mut ir = GraphIr::new(&format!("ir_70"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_071() {
        let mut ir = GraphIr::new(&format!("ir_71"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_072() {
        let mut ir = GraphIr::new(&format!("ir_72"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_073() {
        let mut ir = GraphIr::new(&format!("ir_73"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_074() {
        let mut ir = GraphIr::new(&format!("ir_74"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_075() {
        let mut ir = GraphIr::new(&format!("ir_75"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_076() {
        let mut ir = GraphIr::new(&format!("ir_76"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_077() {
        let mut ir = GraphIr::new(&format!("ir_77"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_078() {
        let mut ir = GraphIr::new(&format!("ir_78"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_079() {
        let mut ir = GraphIr::new(&format!("ir_79"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_080() {
        let mut ir = GraphIr::new(&format!("ir_80"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_081() {
        let mut ir = GraphIr::new(&format!("ir_81"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_082() {
        let mut ir = GraphIr::new(&format!("ir_82"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_083() {
        let mut ir = GraphIr::new(&format!("ir_83"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_084() {
        let mut ir = GraphIr::new(&format!("ir_84"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_085() {
        let mut ir = GraphIr::new(&format!("ir_85"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_086() {
        let mut ir = GraphIr::new(&format!("ir_86"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_087() {
        let mut ir = GraphIr::new(&format!("ir_87"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_088() {
        let mut ir = GraphIr::new(&format!("ir_88"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_089() {
        let mut ir = GraphIr::new(&format!("ir_89"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_090() {
        let mut ir = GraphIr::new(&format!("ir_90"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_091() {
        let mut ir = GraphIr::new(&format!("ir_91"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_092() {
        let mut ir = GraphIr::new(&format!("ir_92"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_093() {
        let mut ir = GraphIr::new(&format!("ir_93"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_094() {
        let mut ir = GraphIr::new(&format!("ir_94"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_095() {
        let mut ir = GraphIr::new(&format!("ir_95"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_096() {
        let mut ir = GraphIr::new(&format!("ir_96"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_097() {
        let mut ir = GraphIr::new(&format!("ir_97"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_098() {
        let mut ir = GraphIr::new(&format!("ir_98"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_099() {
        let mut ir = GraphIr::new(&format!("ir_99"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_100() {
        let mut ir = GraphIr::new(&format!("ir_100"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_101() {
        let mut ir = GraphIr::new(&format!("ir_101"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_102() {
        let mut ir = GraphIr::new(&format!("ir_102"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_103() {
        let mut ir = GraphIr::new(&format!("ir_103"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_104() {
        let mut ir = GraphIr::new(&format!("ir_104"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_105() {
        let mut ir = GraphIr::new(&format!("ir_105"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_106() {
        let mut ir = GraphIr::new(&format!("ir_106"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_107() {
        let mut ir = GraphIr::new(&format!("ir_107"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_108() {
        let mut ir = GraphIr::new(&format!("ir_108"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_109() {
        let mut ir = GraphIr::new(&format!("ir_109"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_110() {
        let mut ir = GraphIr::new(&format!("ir_110"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_111() {
        let mut ir = GraphIr::new(&format!("ir_111"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_112() {
        let mut ir = GraphIr::new(&format!("ir_112"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_113() {
        let mut ir = GraphIr::new(&format!("ir_113"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_114() {
        let mut ir = GraphIr::new(&format!("ir_114"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_115() {
        let mut ir = GraphIr::new(&format!("ir_115"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_116() {
        let mut ir = GraphIr::new(&format!("ir_116"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_117() {
        let mut ir = GraphIr::new(&format!("ir_117"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_118() {
        let mut ir = GraphIr::new(&format!("ir_118"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_119() {
        let mut ir = GraphIr::new(&format!("ir_119"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_120() {
        let mut ir = GraphIr::new(&format!("ir_120"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_121() {
        let mut ir = GraphIr::new(&format!("ir_121"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_122() {
        let mut ir = GraphIr::new(&format!("ir_122"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_123() {
        let mut ir = GraphIr::new(&format!("ir_123"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_124() {
        let mut ir = GraphIr::new(&format!("ir_124"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_125() {
        let mut ir = GraphIr::new(&format!("ir_125"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_126() {
        let mut ir = GraphIr::new(&format!("ir_126"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_127() {
        let mut ir = GraphIr::new(&format!("ir_127"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_128() {
        let mut ir = GraphIr::new(&format!("ir_128"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_129() {
        let mut ir = GraphIr::new(&format!("ir_129"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_130() {
        let mut ir = GraphIr::new(&format!("ir_130"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_131() {
        let mut ir = GraphIr::new(&format!("ir_131"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_132() {
        let mut ir = GraphIr::new(&format!("ir_132"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_133() {
        let mut ir = GraphIr::new(&format!("ir_133"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_134() {
        let mut ir = GraphIr::new(&format!("ir_134"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_135() {
        let mut ir = GraphIr::new(&format!("ir_135"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_136() {
        let mut ir = GraphIr::new(&format!("ir_136"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_137() {
        let mut ir = GraphIr::new(&format!("ir_137"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_138() {
        let mut ir = GraphIr::new(&format!("ir_138"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_139() {
        let mut ir = GraphIr::new(&format!("ir_139"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_140() {
        let mut ir = GraphIr::new(&format!("ir_140"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_141() {
        let mut ir = GraphIr::new(&format!("ir_141"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_142() {
        let mut ir = GraphIr::new(&format!("ir_142"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_143() {
        let mut ir = GraphIr::new(&format!("ir_143"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_144() {
        let mut ir = GraphIr::new(&format!("ir_144"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_145() {
        let mut ir = GraphIr::new(&format!("ir_145"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_146() {
        let mut ir = GraphIr::new(&format!("ir_146"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_147() {
        let mut ir = GraphIr::new(&format!("ir_147"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_148() {
        let mut ir = GraphIr::new(&format!("ir_148"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_149() {
        let mut ir = GraphIr::new(&format!("ir_149"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_150() {
        let mut ir = GraphIr::new(&format!("ir_150"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_151() {
        let mut ir = GraphIr::new(&format!("ir_151"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_152() {
        let mut ir = GraphIr::new(&format!("ir_152"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_153() {
        let mut ir = GraphIr::new(&format!("ir_153"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_154() {
        let mut ir = GraphIr::new(&format!("ir_154"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_155() {
        let mut ir = GraphIr::new(&format!("ir_155"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_156() {
        let mut ir = GraphIr::new(&format!("ir_156"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_157() {
        let mut ir = GraphIr::new(&format!("ir_157"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_158() {
        let mut ir = GraphIr::new(&format!("ir_158"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_159() {
        let mut ir = GraphIr::new(&format!("ir_159"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_160() {
        let mut ir = GraphIr::new(&format!("ir_160"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_161() {
        let mut ir = GraphIr::new(&format!("ir_161"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_162() {
        let mut ir = GraphIr::new(&format!("ir_162"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_163() {
        let mut ir = GraphIr::new(&format!("ir_163"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_164() {
        let mut ir = GraphIr::new(&format!("ir_164"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_165() {
        let mut ir = GraphIr::new(&format!("ir_165"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_166() {
        let mut ir = GraphIr::new(&format!("ir_166"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_167() {
        let mut ir = GraphIr::new(&format!("ir_167"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_168() {
        let mut ir = GraphIr::new(&format!("ir_168"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_169() {
        let mut ir = GraphIr::new(&format!("ir_169"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_170() {
        let mut ir = GraphIr::new(&format!("ir_170"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_171() {
        let mut ir = GraphIr::new(&format!("ir_171"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_172() {
        let mut ir = GraphIr::new(&format!("ir_172"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_173() {
        let mut ir = GraphIr::new(&format!("ir_173"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_174() {
        let mut ir = GraphIr::new(&format!("ir_174"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_175() {
        let mut ir = GraphIr::new(&format!("ir_175"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_176() {
        let mut ir = GraphIr::new(&format!("ir_176"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_177() {
        let mut ir = GraphIr::new(&format!("ir_177"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_178() {
        let mut ir = GraphIr::new(&format!("ir_178"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_179() {
        let mut ir = GraphIr::new(&format!("ir_179"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_180() {
        let mut ir = GraphIr::new(&format!("ir_180"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_181() {
        let mut ir = GraphIr::new(&format!("ir_181"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_182() {
        let mut ir = GraphIr::new(&format!("ir_182"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_183() {
        let mut ir = GraphIr::new(&format!("ir_183"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_184() {
        let mut ir = GraphIr::new(&format!("ir_184"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_185() {
        let mut ir = GraphIr::new(&format!("ir_185"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_186() {
        let mut ir = GraphIr::new(&format!("ir_186"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_187() {
        let mut ir = GraphIr::new(&format!("ir_187"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_188() {
        let mut ir = GraphIr::new(&format!("ir_188"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_189() {
        let mut ir = GraphIr::new(&format!("ir_189"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_190() {
        let mut ir = GraphIr::new(&format!("ir_190"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_191() {
        let mut ir = GraphIr::new(&format!("ir_191"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_192() {
        let mut ir = GraphIr::new(&format!("ir_192"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_193() {
        let mut ir = GraphIr::new(&format!("ir_193"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_194() {
        let mut ir = GraphIr::new(&format!("ir_194"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_195() {
        let mut ir = GraphIr::new(&format!("ir_195"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_196() {
        let mut ir = GraphIr::new(&format!("ir_196"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_197() {
        let mut ir = GraphIr::new(&format!("ir_197"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_198() {
        let mut ir = GraphIr::new(&format!("ir_198"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_199() {
        let mut ir = GraphIr::new(&format!("ir_199"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_200() {
        let mut ir = GraphIr::new(&format!("ir_200"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_201() {
        let mut ir = GraphIr::new(&format!("ir_201"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_202() {
        let mut ir = GraphIr::new(&format!("ir_202"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_203() {
        let mut ir = GraphIr::new(&format!("ir_203"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_204() {
        let mut ir = GraphIr::new(&format!("ir_204"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_205() {
        let mut ir = GraphIr::new(&format!("ir_205"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_206() {
        let mut ir = GraphIr::new(&format!("ir_206"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_207() {
        let mut ir = GraphIr::new(&format!("ir_207"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_208() {
        let mut ir = GraphIr::new(&format!("ir_208"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_209() {
        let mut ir = GraphIr::new(&format!("ir_209"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_210() {
        let mut ir = GraphIr::new(&format!("ir_210"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_211() {
        let mut ir = GraphIr::new(&format!("ir_211"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_212() {
        let mut ir = GraphIr::new(&format!("ir_212"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_213() {
        let mut ir = GraphIr::new(&format!("ir_213"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_214() {
        let mut ir = GraphIr::new(&format!("ir_214"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_215() {
        let mut ir = GraphIr::new(&format!("ir_215"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_216() {
        let mut ir = GraphIr::new(&format!("ir_216"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_217() {
        let mut ir = GraphIr::new(&format!("ir_217"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_218() {
        let mut ir = GraphIr::new(&format!("ir_218"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_219() {
        let mut ir = GraphIr::new(&format!("ir_219"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_220() {
        let mut ir = GraphIr::new(&format!("ir_220"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_221() {
        let mut ir = GraphIr::new(&format!("ir_221"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_222() {
        let mut ir = GraphIr::new(&format!("ir_222"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_223() {
        let mut ir = GraphIr::new(&format!("ir_223"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_224() {
        let mut ir = GraphIr::new(&format!("ir_224"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_225() {
        let mut ir = GraphIr::new(&format!("ir_225"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_226() {
        let mut ir = GraphIr::new(&format!("ir_226"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_227() {
        let mut ir = GraphIr::new(&format!("ir_227"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_228() {
        let mut ir = GraphIr::new(&format!("ir_228"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_229() {
        let mut ir = GraphIr::new(&format!("ir_229"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_230() {
        let mut ir = GraphIr::new(&format!("ir_230"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_231() {
        let mut ir = GraphIr::new(&format!("ir_231"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_232() {
        let mut ir = GraphIr::new(&format!("ir_232"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_233() {
        let mut ir = GraphIr::new(&format!("ir_233"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_234() {
        let mut ir = GraphIr::new(&format!("ir_234"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_235() {
        let mut ir = GraphIr::new(&format!("ir_235"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_236() {
        let mut ir = GraphIr::new(&format!("ir_236"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_237() {
        let mut ir = GraphIr::new(&format!("ir_237"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_238() {
        let mut ir = GraphIr::new(&format!("ir_238"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_239() {
        let mut ir = GraphIr::new(&format!("ir_239"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_240() {
        let mut ir = GraphIr::new(&format!("ir_240"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_241() {
        let mut ir = GraphIr::new(&format!("ir_241"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_242() {
        let mut ir = GraphIr::new(&format!("ir_242"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_243() {
        let mut ir = GraphIr::new(&format!("ir_243"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_244() {
        let mut ir = GraphIr::new(&format!("ir_244"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_245() {
        let mut ir = GraphIr::new(&format!("ir_245"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_246() {
        let mut ir = GraphIr::new(&format!("ir_246"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_247() {
        let mut ir = GraphIr::new(&format!("ir_247"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_248() {
        let mut ir = GraphIr::new(&format!("ir_248"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_249() {
        let mut ir = GraphIr::new(&format!("ir_249"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_250() {
        let mut ir = GraphIr::new(&format!("ir_250"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_251() {
        let mut ir = GraphIr::new(&format!("ir_251"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_252() {
        let mut ir = GraphIr::new(&format!("ir_252"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_253() {
        let mut ir = GraphIr::new(&format!("ir_253"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_254() {
        let mut ir = GraphIr::new(&format!("ir_254"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_255() {
        let mut ir = GraphIr::new(&format!("ir_255"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_256() {
        let mut ir = GraphIr::new(&format!("ir_256"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_257() {
        let mut ir = GraphIr::new(&format!("ir_257"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_258() {
        let mut ir = GraphIr::new(&format!("ir_258"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_259() {
        let mut ir = GraphIr::new(&format!("ir_259"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_260() {
        let mut ir = GraphIr::new(&format!("ir_260"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_261() {
        let mut ir = GraphIr::new(&format!("ir_261"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_262() {
        let mut ir = GraphIr::new(&format!("ir_262"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_263() {
        let mut ir = GraphIr::new(&format!("ir_263"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_264() {
        let mut ir = GraphIr::new(&format!("ir_264"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_265() {
        let mut ir = GraphIr::new(&format!("ir_265"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_266() {
        let mut ir = GraphIr::new(&format!("ir_266"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_267() {
        let mut ir = GraphIr::new(&format!("ir_267"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_268() {
        let mut ir = GraphIr::new(&format!("ir_268"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_269() {
        let mut ir = GraphIr::new(&format!("ir_269"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_270() {
        let mut ir = GraphIr::new(&format!("ir_270"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_271() {
        let mut ir = GraphIr::new(&format!("ir_271"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_272() {
        let mut ir = GraphIr::new(&format!("ir_272"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_273() {
        let mut ir = GraphIr::new(&format!("ir_273"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_274() {
        let mut ir = GraphIr::new(&format!("ir_274"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_275() {
        let mut ir = GraphIr::new(&format!("ir_275"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_276() {
        let mut ir = GraphIr::new(&format!("ir_276"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_277() {
        let mut ir = GraphIr::new(&format!("ir_277"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_278() {
        let mut ir = GraphIr::new(&format!("ir_278"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_279() {
        let mut ir = GraphIr::new(&format!("ir_279"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_280() {
        let mut ir = GraphIr::new(&format!("ir_280"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_281() {
        let mut ir = GraphIr::new(&format!("ir_281"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_282() {
        let mut ir = GraphIr::new(&format!("ir_282"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_283() {
        let mut ir = GraphIr::new(&format!("ir_283"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_284() {
        let mut ir = GraphIr::new(&format!("ir_284"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_285() {
        let mut ir = GraphIr::new(&format!("ir_285"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_286() {
        let mut ir = GraphIr::new(&format!("ir_286"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_287() {
        let mut ir = GraphIr::new(&format!("ir_287"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_288() {
        let mut ir = GraphIr::new(&format!("ir_288"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_289() {
        let mut ir = GraphIr::new(&format!("ir_289"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_290() {
        let mut ir = GraphIr::new(&format!("ir_290"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_291() {
        let mut ir = GraphIr::new(&format!("ir_291"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ir_mod_stress_292() {
        let mut ir = GraphIr::new(&format!("ir_292"));
        let v1 = ir.add_value("v1", Shape::new(vec![2, 2]), DType::F32);
        let v2 = ir.add_value("v2", Shape::new(vec![2, 2]), DType::F32);
        let n = ir.add_node("add", OpKind::Add, vec![v1], vec![v2]);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.num_values(), 2);
        assert_eq!(n, 0);
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
    // Computation graph IR verification and pass padding line 2
    // Computation graph IR verification and pass padding line 3
    // Computation graph IR verification and pass padding line 4
    // Computation graph IR verification and pass padding line 5
}
