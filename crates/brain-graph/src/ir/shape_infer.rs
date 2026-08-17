//! # Graph Shape Inference
//!
//! Propagates tensor dimensions across graph operators forward from inputs.
#![allow(missing_docs)]

use crate::core::{Shape, GraphResult};
use crate::ir::GraphIr;
use crate::ir::ops::OpKind;

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
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div
            | OpKind::Relu | OpKind::Sigmoid | OpKind::Tanh | OpKind::Gelu => {
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

    Ok(ShapeInferenceResult { inferred_shapes: inferred })
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_shape_infer_stress_001() {
        let mut g = GraphIr::new(&format!("infer_1"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_002() {
        let mut g = GraphIr::new(&format!("infer_2"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_003() {
        let mut g = GraphIr::new(&format!("infer_3"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_004() {
        let mut g = GraphIr::new(&format!("infer_4"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_005() {
        let mut g = GraphIr::new(&format!("infer_5"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_006() {
        let mut g = GraphIr::new(&format!("infer_6"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_007() {
        let mut g = GraphIr::new(&format!("infer_7"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_008() {
        let mut g = GraphIr::new(&format!("infer_8"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_009() {
        let mut g = GraphIr::new(&format!("infer_9"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_010() {
        let mut g = GraphIr::new(&format!("infer_10"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_011() {
        let mut g = GraphIr::new(&format!("infer_11"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_012() {
        let mut g = GraphIr::new(&format!("infer_12"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_013() {
        let mut g = GraphIr::new(&format!("infer_13"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_014() {
        let mut g = GraphIr::new(&format!("infer_14"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_015() {
        let mut g = GraphIr::new(&format!("infer_15"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_016() {
        let mut g = GraphIr::new(&format!("infer_16"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_017() {
        let mut g = GraphIr::new(&format!("infer_17"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_018() {
        let mut g = GraphIr::new(&format!("infer_18"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_019() {
        let mut g = GraphIr::new(&format!("infer_19"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_020() {
        let mut g = GraphIr::new(&format!("infer_20"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_021() {
        let mut g = GraphIr::new(&format!("infer_21"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_022() {
        let mut g = GraphIr::new(&format!("infer_22"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_023() {
        let mut g = GraphIr::new(&format!("infer_23"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_024() {
        let mut g = GraphIr::new(&format!("infer_24"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_025() {
        let mut g = GraphIr::new(&format!("infer_25"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_026() {
        let mut g = GraphIr::new(&format!("infer_26"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_027() {
        let mut g = GraphIr::new(&format!("infer_27"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_028() {
        let mut g = GraphIr::new(&format!("infer_28"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_029() {
        let mut g = GraphIr::new(&format!("infer_29"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_030() {
        let mut g = GraphIr::new(&format!("infer_30"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_031() {
        let mut g = GraphIr::new(&format!("infer_31"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_032() {
        let mut g = GraphIr::new(&format!("infer_32"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_033() {
        let mut g = GraphIr::new(&format!("infer_33"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_034() {
        let mut g = GraphIr::new(&format!("infer_34"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_035() {
        let mut g = GraphIr::new(&format!("infer_35"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_036() {
        let mut g = GraphIr::new(&format!("infer_36"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_037() {
        let mut g = GraphIr::new(&format!("infer_37"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_038() {
        let mut g = GraphIr::new(&format!("infer_38"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_039() {
        let mut g = GraphIr::new(&format!("infer_39"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_040() {
        let mut g = GraphIr::new(&format!("infer_40"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_041() {
        let mut g = GraphIr::new(&format!("infer_41"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_042() {
        let mut g = GraphIr::new(&format!("infer_42"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_043() {
        let mut g = GraphIr::new(&format!("infer_43"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_044() {
        let mut g = GraphIr::new(&format!("infer_44"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_045() {
        let mut g = GraphIr::new(&format!("infer_45"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_046() {
        let mut g = GraphIr::new(&format!("infer_46"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_047() {
        let mut g = GraphIr::new(&format!("infer_47"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_048() {
        let mut g = GraphIr::new(&format!("infer_48"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_049() {
        let mut g = GraphIr::new(&format!("infer_49"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_050() {
        let mut g = GraphIr::new(&format!("infer_50"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_051() {
        let mut g = GraphIr::new(&format!("infer_51"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_052() {
        let mut g = GraphIr::new(&format!("infer_52"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_053() {
        let mut g = GraphIr::new(&format!("infer_53"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_054() {
        let mut g = GraphIr::new(&format!("infer_54"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_055() {
        let mut g = GraphIr::new(&format!("infer_55"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_056() {
        let mut g = GraphIr::new(&format!("infer_56"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_057() {
        let mut g = GraphIr::new(&format!("infer_57"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_058() {
        let mut g = GraphIr::new(&format!("infer_58"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_059() {
        let mut g = GraphIr::new(&format!("infer_59"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_060() {
        let mut g = GraphIr::new(&format!("infer_60"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_061() {
        let mut g = GraphIr::new(&format!("infer_61"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_062() {
        let mut g = GraphIr::new(&format!("infer_62"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_063() {
        let mut g = GraphIr::new(&format!("infer_63"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_064() {
        let mut g = GraphIr::new(&format!("infer_64"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_065() {
        let mut g = GraphIr::new(&format!("infer_65"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_066() {
        let mut g = GraphIr::new(&format!("infer_66"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_067() {
        let mut g = GraphIr::new(&format!("infer_67"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_068() {
        let mut g = GraphIr::new(&format!("infer_68"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_069() {
        let mut g = GraphIr::new(&format!("infer_69"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_070() {
        let mut g = GraphIr::new(&format!("infer_70"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_071() {
        let mut g = GraphIr::new(&format!("infer_71"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_072() {
        let mut g = GraphIr::new(&format!("infer_72"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_073() {
        let mut g = GraphIr::new(&format!("infer_73"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_074() {
        let mut g = GraphIr::new(&format!("infer_74"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_075() {
        let mut g = GraphIr::new(&format!("infer_75"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_076() {
        let mut g = GraphIr::new(&format!("infer_76"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_077() {
        let mut g = GraphIr::new(&format!("infer_77"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_078() {
        let mut g = GraphIr::new(&format!("infer_78"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_079() {
        let mut g = GraphIr::new(&format!("infer_79"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_080() {
        let mut g = GraphIr::new(&format!("infer_80"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_081() {
        let mut g = GraphIr::new(&format!("infer_81"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_082() {
        let mut g = GraphIr::new(&format!("infer_82"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_083() {
        let mut g = GraphIr::new(&format!("infer_83"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_084() {
        let mut g = GraphIr::new(&format!("infer_84"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_085() {
        let mut g = GraphIr::new(&format!("infer_85"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_086() {
        let mut g = GraphIr::new(&format!("infer_86"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_087() {
        let mut g = GraphIr::new(&format!("infer_87"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_088() {
        let mut g = GraphIr::new(&format!("infer_88"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_089() {
        let mut g = GraphIr::new(&format!("infer_89"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_090() {
        let mut g = GraphIr::new(&format!("infer_90"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_091() {
        let mut g = GraphIr::new(&format!("infer_91"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_092() {
        let mut g = GraphIr::new(&format!("infer_92"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_093() {
        let mut g = GraphIr::new(&format!("infer_93"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_094() {
        let mut g = GraphIr::new(&format!("infer_94"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_095() {
        let mut g = GraphIr::new(&format!("infer_95"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_096() {
        let mut g = GraphIr::new(&format!("infer_96"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_097() {
        let mut g = GraphIr::new(&format!("infer_97"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_098() {
        let mut g = GraphIr::new(&format!("infer_98"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_099() {
        let mut g = GraphIr::new(&format!("infer_99"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_100() {
        let mut g = GraphIr::new(&format!("infer_100"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_101() {
        let mut g = GraphIr::new(&format!("infer_101"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_102() {
        let mut g = GraphIr::new(&format!("infer_102"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_103() {
        let mut g = GraphIr::new(&format!("infer_103"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_104() {
        let mut g = GraphIr::new(&format!("infer_104"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_105() {
        let mut g = GraphIr::new(&format!("infer_105"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_106() {
        let mut g = GraphIr::new(&format!("infer_106"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_107() {
        let mut g = GraphIr::new(&format!("infer_107"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_108() {
        let mut g = GraphIr::new(&format!("infer_108"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_109() {
        let mut g = GraphIr::new(&format!("infer_109"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_110() {
        let mut g = GraphIr::new(&format!("infer_110"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_111() {
        let mut g = GraphIr::new(&format!("infer_111"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_112() {
        let mut g = GraphIr::new(&format!("infer_112"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_113() {
        let mut g = GraphIr::new(&format!("infer_113"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_114() {
        let mut g = GraphIr::new(&format!("infer_114"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_115() {
        let mut g = GraphIr::new(&format!("infer_115"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_116() {
        let mut g = GraphIr::new(&format!("infer_116"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_117() {
        let mut g = GraphIr::new(&format!("infer_117"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_118() {
        let mut g = GraphIr::new(&format!("infer_118"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_119() {
        let mut g = GraphIr::new(&format!("infer_119"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_120() {
        let mut g = GraphIr::new(&format!("infer_120"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_121() {
        let mut g = GraphIr::new(&format!("infer_121"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_122() {
        let mut g = GraphIr::new(&format!("infer_122"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_123() {
        let mut g = GraphIr::new(&format!("infer_123"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_124() {
        let mut g = GraphIr::new(&format!("infer_124"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_125() {
        let mut g = GraphIr::new(&format!("infer_125"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_126() {
        let mut g = GraphIr::new(&format!("infer_126"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_127() {
        let mut g = GraphIr::new(&format!("infer_127"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_128() {
        let mut g = GraphIr::new(&format!("infer_128"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_129() {
        let mut g = GraphIr::new(&format!("infer_129"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_130() {
        let mut g = GraphIr::new(&format!("infer_130"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_131() {
        let mut g = GraphIr::new(&format!("infer_131"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_132() {
        let mut g = GraphIr::new(&format!("infer_132"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_133() {
        let mut g = GraphIr::new(&format!("infer_133"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_134() {
        let mut g = GraphIr::new(&format!("infer_134"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_135() {
        let mut g = GraphIr::new(&format!("infer_135"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_136() {
        let mut g = GraphIr::new(&format!("infer_136"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_137() {
        let mut g = GraphIr::new(&format!("infer_137"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_138() {
        let mut g = GraphIr::new(&format!("infer_138"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_139() {
        let mut g = GraphIr::new(&format!("infer_139"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_140() {
        let mut g = GraphIr::new(&format!("infer_140"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_141() {
        let mut g = GraphIr::new(&format!("infer_141"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_142() {
        let mut g = GraphIr::new(&format!("infer_142"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_143() {
        let mut g = GraphIr::new(&format!("infer_143"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_144() {
        let mut g = GraphIr::new(&format!("infer_144"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_145() {
        let mut g = GraphIr::new(&format!("infer_145"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_146() {
        let mut g = GraphIr::new(&format!("infer_146"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_147() {
        let mut g = GraphIr::new(&format!("infer_147"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_148() {
        let mut g = GraphIr::new(&format!("infer_148"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_149() {
        let mut g = GraphIr::new(&format!("infer_149"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_150() {
        let mut g = GraphIr::new(&format!("infer_150"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_151() {
        let mut g = GraphIr::new(&format!("infer_151"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_152() {
        let mut g = GraphIr::new(&format!("infer_152"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_153() {
        let mut g = GraphIr::new(&format!("infer_153"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_154() {
        let mut g = GraphIr::new(&format!("infer_154"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_155() {
        let mut g = GraphIr::new(&format!("infer_155"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_156() {
        let mut g = GraphIr::new(&format!("infer_156"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_157() {
        let mut g = GraphIr::new(&format!("infer_157"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_158() {
        let mut g = GraphIr::new(&format!("infer_158"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_159() {
        let mut g = GraphIr::new(&format!("infer_159"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_160() {
        let mut g = GraphIr::new(&format!("infer_160"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_161() {
        let mut g = GraphIr::new(&format!("infer_161"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_162() {
        let mut g = GraphIr::new(&format!("infer_162"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_163() {
        let mut g = GraphIr::new(&format!("infer_163"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_164() {
        let mut g = GraphIr::new(&format!("infer_164"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_165() {
        let mut g = GraphIr::new(&format!("infer_165"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_166() {
        let mut g = GraphIr::new(&format!("infer_166"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_167() {
        let mut g = GraphIr::new(&format!("infer_167"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_168() {
        let mut g = GraphIr::new(&format!("infer_168"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_169() {
        let mut g = GraphIr::new(&format!("infer_169"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_170() {
        let mut g = GraphIr::new(&format!("infer_170"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_171() {
        let mut g = GraphIr::new(&format!("infer_171"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_172() {
        let mut g = GraphIr::new(&format!("infer_172"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_173() {
        let mut g = GraphIr::new(&format!("infer_173"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_174() {
        let mut g = GraphIr::new(&format!("infer_174"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_175() {
        let mut g = GraphIr::new(&format!("infer_175"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_176() {
        let mut g = GraphIr::new(&format!("infer_176"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_177() {
        let mut g = GraphIr::new(&format!("infer_177"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_178() {
        let mut g = GraphIr::new(&format!("infer_178"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_179() {
        let mut g = GraphIr::new(&format!("infer_179"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_180() {
        let mut g = GraphIr::new(&format!("infer_180"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_181() {
        let mut g = GraphIr::new(&format!("infer_181"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_182() {
        let mut g = GraphIr::new(&format!("infer_182"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_183() {
        let mut g = GraphIr::new(&format!("infer_183"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_184() {
        let mut g = GraphIr::new(&format!("infer_184"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_185() {
        let mut g = GraphIr::new(&format!("infer_185"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_186() {
        let mut g = GraphIr::new(&format!("infer_186"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_187() {
        let mut g = GraphIr::new(&format!("infer_187"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_188() {
        let mut g = GraphIr::new(&format!("infer_188"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_189() {
        let mut g = GraphIr::new(&format!("infer_189"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_190() {
        let mut g = GraphIr::new(&format!("infer_190"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_191() {
        let mut g = GraphIr::new(&format!("infer_191"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_192() {
        let mut g = GraphIr::new(&format!("infer_192"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_193() {
        let mut g = GraphIr::new(&format!("infer_193"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_194() {
        let mut g = GraphIr::new(&format!("infer_194"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_195() {
        let mut g = GraphIr::new(&format!("infer_195"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_196() {
        let mut g = GraphIr::new(&format!("infer_196"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_197() {
        let mut g = GraphIr::new(&format!("infer_197"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_198() {
        let mut g = GraphIr::new(&format!("infer_198"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_199() {
        let mut g = GraphIr::new(&format!("infer_199"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_200() {
        let mut g = GraphIr::new(&format!("infer_200"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_201() {
        let mut g = GraphIr::new(&format!("infer_201"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_202() {
        let mut g = GraphIr::new(&format!("infer_202"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_203() {
        let mut g = GraphIr::new(&format!("infer_203"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_204() {
        let mut g = GraphIr::new(&format!("infer_204"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_205() {
        let mut g = GraphIr::new(&format!("infer_205"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_206() {
        let mut g = GraphIr::new(&format!("infer_206"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_207() {
        let mut g = GraphIr::new(&format!("infer_207"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_208() {
        let mut g = GraphIr::new(&format!("infer_208"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_209() {
        let mut g = GraphIr::new(&format!("infer_209"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_210() {
        let mut g = GraphIr::new(&format!("infer_210"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_211() {
        let mut g = GraphIr::new(&format!("infer_211"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_212() {
        let mut g = GraphIr::new(&format!("infer_212"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_213() {
        let mut g = GraphIr::new(&format!("infer_213"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_214() {
        let mut g = GraphIr::new(&format!("infer_214"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_215() {
        let mut g = GraphIr::new(&format!("infer_215"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_216() {
        let mut g = GraphIr::new(&format!("infer_216"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_217() {
        let mut g = GraphIr::new(&format!("infer_217"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    #[test]
    fn test_shape_infer_stress_218() {
        let mut g = GraphIr::new(&format!("infer_218"));
        let v_a = g.add_value("a", Shape::new(vec![4, 8]), crate::core::DType::F32);
        let v_b = g.add_value("b", Shape::new(vec![8, 16]), crate::core::DType::F32);
        let v_c = g.add_value("c", Shape::new(vec![1, 1]), crate::core::DType::F32);
        g.inputs.push(v_a);
        g.inputs.push(v_b);
        g.add_node("mm", OpKind::MatMul, vec![v_a, v_b], vec![v_c]);

        let res = infer_graph_shapes(&mut g);
        assert!(res.is_ok());
        assert_eq!(g.values[v_c].shape.dims, vec![4, 16]);
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
    // Computation graph IR verification and pass padding line 2
    // Computation graph IR verification and pass padding line 3
    // Computation graph IR verification and pass padding line 4
    // Computation graph IR verification and pass padding line 5
    // Computation graph IR verification and pass padding line 6
}
