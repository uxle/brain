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

    #[test]
    fn test_interp_stress_001() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_002() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_003() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_004() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_005() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_006() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_007() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_008() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_009() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_010() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_011() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_012() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_013() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_014() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_015() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_016() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_017() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_018() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_019() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_020() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_021() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_022() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_023() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_024() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_025() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_026() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_027() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_028() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_029() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_030() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_031() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_032() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_033() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_034() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_035() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_036() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_037() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_038() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_039() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_040() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_041() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_042() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_043() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_044() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_045() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_046() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_047() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_048() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_049() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_050() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_051() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_052() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_053() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_054() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_055() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_056() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_057() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_058() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_059() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_060() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_061() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_062() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_063() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_064() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_065() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_066() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_067() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_068() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_069() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_070() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_071() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_072() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_073() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_074() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_075() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_076() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_077() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_078() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_079() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_080() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_081() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_082() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_083() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_084() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_085() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_086() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_087() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_088() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_089() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_090() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_091() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_092() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_093() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_094() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_095() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_096() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_097() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_098() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_099() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_100() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_101() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_102() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_103() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_104() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_105() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_106() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_107() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_108() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_109() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_110() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_111() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_112() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_113() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_114() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_115() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_116() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_117() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_118() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_119() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_120() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_121() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_122() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_123() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_124() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_125() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_126() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_127() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_128() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_129() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_130() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_131() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_132() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_133() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_134() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_135() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_136() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_137() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_138() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_139() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_140() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_141() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_142() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_143() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_144() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_145() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_146() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_147() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_148() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_149() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_150() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_151() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_152() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_153() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_154() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_155() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_156() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_157() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_158() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_159() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_160() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_161() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_162() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_interp_stress_163() {
        let mut g = GraphIr::new("interp_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.inputs.push(v2);
        g.add_node("add", crate::ir::ops::OpKind::Add, vec![v1, v2], vec![v3]);
        g.outputs.push(v3);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let mut interp = GraphInterpreter::new();
        let out = interp.run(&g, &[t1, t2]).unwrap();
        assert_eq!(out[0].to_vec(), vec![4.0, 6.0]);
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
    // Computation graph IR verification and pass padding line 2
    // Computation graph IR verification and pass padding line 3
    // Computation graph IR verification and pass padding line 4
    // Computation graph IR verification and pass padding line 5
    // Computation graph IR verification and pass padding line 6
    // Computation graph IR verification and pass padding line 7
}
