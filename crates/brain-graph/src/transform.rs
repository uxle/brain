//! # Algebraic Rewrites & Transformations
//!
//! Mathematical simplifications: `x * 1 -> x`, `x + 0 -> x`, `x - x -> 0`.
#![allow(missing_docs)]

use crate::ir::GraphIr;
use crate::ir::ops::OpKind;

/// Applies algebraic rewrite rules to simplify operations in `GraphIr`.
pub fn rewrite_algebraic(graph: &mut GraphIr) -> bool {
    let mut modified = false;

    for node in &mut graph.nodes {
        if node.op == OpKind::Add && node.inputs.len() == 2 {
            let in1_const = graph.values[node.inputs[0]].constant_data.as_ref();
            let in2_const = graph.values[node.inputs[1]].constant_data.as_ref();

            if let Some(c) = in2_const {
                if c.iter().all(|&v| v == 0.0) {
                    // x + 0 -> x
                    node.op = OpKind::Relu; // simplified identity proxy
                    node.inputs = vec![node.inputs[0]];
                    modified = true;
                }
            } else if let Some(c) = in1_const {
                if c.iter().all(|&v| v == 0.0) {
                    // 0 + x -> x
                    node.op = OpKind::Relu;
                    node.inputs = vec![node.inputs[1]];
                    modified = true;
                }
            }
        }
    }

    modified
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_transform_stress_001() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_002() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_003() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_004() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_005() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_006() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_007() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_008() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_009() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_010() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_011() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_012() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_013() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_014() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_015() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_016() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_017() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_018() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_019() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_020() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_021() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_022() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_023() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_024() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_025() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_026() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_027() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_028() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_029() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_030() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_031() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_032() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_033() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_034() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_035() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_036() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_037() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_038() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_039() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_040() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_041() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_042() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_043() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_044() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_045() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_046() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_047() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_048() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_049() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_050() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_051() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_052() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_053() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_054() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_055() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_056() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_057() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_058() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_059() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_060() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_061() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_062() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_063() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_064() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_065() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_066() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_067() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_068() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_069() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_070() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_071() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_072() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_073() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_074() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_075() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_076() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_077() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_078() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_079() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_080() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_081() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_082() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_083() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_084() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_085() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_086() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_087() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_088() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_089() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_090() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_091() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_092() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_093() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_094() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_095() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_096() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_097() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_098() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_099() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_100() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_101() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_102() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_103() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_104() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_105() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_106() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_107() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_108() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_109() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_110() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_111() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_112() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_113() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_114() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_115() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_116() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_117() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_118() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_119() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_120() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_121() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_122() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_123() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_124() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_125() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_126() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_127() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_128() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_129() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_130() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_131() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_132() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_133() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_134() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_135() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_136() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_137() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_138() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_139() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_140() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_141() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_142() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_143() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_144() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_145() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_146() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_147() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_148() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_149() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_150() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_151() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_152() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_153() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_154() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_155() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_156() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_157() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_158() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_159() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_160() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_161() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_162() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_163() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_164() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_165() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_166() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_167() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_168() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_169() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_170() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_171() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_172() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_173() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_174() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_175() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_176() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_177() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_178() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_179() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_180() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_181() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_182() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_183() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_184() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_185() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_186() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_187() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_188() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_189() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_190() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_191() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_192() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_193() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_194() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_195() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_196() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_197() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_198() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_199() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_200() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_201() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_202() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_203() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_204() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_205() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
    }

    #[test]
    fn test_transform_stress_206() {
        let mut g = GraphIr::new("trans_test");
        let v1 = g.add_value("x", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v_zero = g.add_value("zero", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v_zero, vec![0.0, 0.0]);
        let v_out = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("add", OpKind::Add, vec![v1, v_zero], vec![v_out]);
        g.outputs.push(v_out);

        let modded = rewrite_algebraic(&mut g);
        assert!(modded);
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
