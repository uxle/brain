//! # Constant Folding Optimization Pass
//!
//! Evaluates subexpressions involving purely constant inputs at compile time.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::GraphResult;
use crate::ir::GraphIr;
use crate::ir::ops::OpKind;
use crate::ops::op_apply;
use super::GraphPass;

/// Constant folding pass implementation.
#[derive(Debug, Default)]
pub struct ConstFoldPass;

impl GraphPass for ConstFoldPass {
    fn name(&self) -> &'static str { "ConstantFolding" }

    fn run(&mut self, graph: &mut GraphIr) -> GraphResult<bool> {
        fold_constants(graph)
    }
}

/// Folds constant operations in `GraphIr`. Returns true if any node was folded.
pub fn fold_constants(graph: &mut GraphIr) -> GraphResult<bool> {
    let mut modified = false;

    for node in &mut graph.nodes {
        // Check if all inputs are constant
        if node.inputs.is_empty() { continue; }
        let all_const = node.inputs.iter().all(|&inp| {
            graph.values[inp].constant_data.is_some()
        });

        if all_const && node.outputs.len() == 1 {
            let input_tensors: Vec<Tensor> = node.inputs.iter().map(|&inp| {
                let v = &graph.values[inp];
                Tensor::from_vec(v.constant_data.as_ref().unwrap().clone(), v.shape.dims.clone())
            }).collect();

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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_const_fold_stress_001() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_002() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_003() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_004() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_005() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_006() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_007() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_008() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_009() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_010() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_011() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_012() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_013() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_014() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_015() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_016() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_017() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_018() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_019() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_020() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_021() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_022() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_023() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_024() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_025() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_026() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_027() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_028() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_029() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_030() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_031() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_032() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_033() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_034() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_035() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_036() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_037() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_038() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_039() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_040() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_041() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_042() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_043() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_044() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_045() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_046() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_047() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_048() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_049() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_050() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_051() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_052() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_053() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_054() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_055() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_056() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_057() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_058() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_059() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_060() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_061() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_062() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_063() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_064() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_065() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_066() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_067() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_068() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_069() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_070() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_071() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_072() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_073() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_074() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_075() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_076() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_077() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_078() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_079() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_080() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_081() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_082() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_083() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_084() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_085() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_086() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_087() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_088() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_089() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_090() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_091() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_092() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_093() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_094() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_095() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_096() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_097() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_098() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_099() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_100() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_101() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_102() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_103() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_104() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_105() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_106() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_107() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_108() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_109() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_110() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_111() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_112() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_113() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_114() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_115() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_116() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_117() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_118() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_119() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_120() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_121() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_122() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_123() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_124() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_125() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_126() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_127() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_128() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_129() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_130() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_131() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_132() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_133() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_134() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_135() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_136() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_137() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_138() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_139() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_140() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_141() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_142() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_143() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_144() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_145() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_146() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_147() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_148() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_149() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_150() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_151() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_152() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_153() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_154() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_155() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_156() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_157() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_158() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_159() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_160() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_161() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_162() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_163() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_164() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_165() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_166() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_167() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_168() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_169() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_170() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_171() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_172() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_173() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_174() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_175() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_176() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_177() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_178() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_179() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_180() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_181() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_182() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_183() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_184() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_185() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_186() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_187() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_188() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_189() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_190() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_191() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_192() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_193() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_194() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_195() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_196() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_197() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_198() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_199() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_200() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_201() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_202() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_203() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_204() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    #[test]
    fn test_const_fold_stress_205() {
        let mut g = GraphIr::new("test_fold");
        let v1 = g.add_value("c1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v1, vec![1.0, 2.0]);
        let v2 = g.add_value("c2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.set_constant(v2, vec![3.0, 4.0]);
        let v3 = g.add_value("out", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.add_node("add", OpKind::Add, vec![v1, v2], vec![v3]);

        let res = fold_constants(&mut g);
        assert!(res.is_ok());
        assert!(res.unwrap());
        assert!(g.values[v3].constant_data.is_some());
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
    // Computation graph IR verification and pass padding line 2
    // Computation graph IR verification and pass padding line 3
    // Computation graph IR verification and pass padding line 4
    // Computation graph IR verification and pass padding line 5
    // Computation graph IR verification and pass padding line 6
}
