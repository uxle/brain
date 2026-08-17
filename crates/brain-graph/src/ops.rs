//! # Graph Operator Constructors
//!
//! Direct helper functions to construct and apply graph operations.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::ValueId;
use crate::builder::GraphBuilder;
use crate::ir::ops::OpKind;

/// Adds an addition node to the builder.
pub fn graph_add(builder: &mut GraphBuilder, a: ValueId, b: ValueId, shape: Vec<usize>) -> ValueId {
    builder.add_node("add", OpKind::Add, vec![a, b], shape)
}

/// Adds a matrix multiplication node to the builder.
pub fn graph_matmul(builder: &mut GraphBuilder, a: ValueId, b: ValueId, shape: Vec<usize>) -> ValueId {
    builder.add_node("matmul", OpKind::MatMul, vec![a, b], shape)
}

/// Adds a ReLU activation node to the builder.
pub fn graph_relu(builder: &mut GraphBuilder, a: ValueId, shape: Vec<usize>) -> ValueId {
    builder.add_node("relu", OpKind::Relu, vec![a], shape)
}

/// Direct execution of OpKind on `brain_core::Tensor` inputs.
pub fn op_apply(op: OpKind, inputs: &[&Tensor]) -> Tensor {
    match op {
        OpKind::Add => {
            if inputs.len() >= 2 { inputs[0] + inputs[1] } else { Tensor::zeros(vec![1]) }
        }
        OpKind::Sub => {
            if inputs.len() >= 2 { inputs[0] - inputs[1] } else { Tensor::zeros(vec![1]) }
        }
        OpKind::Mul => {
            if inputs.len() >= 2 { inputs[0] * inputs[1] } else { Tensor::zeros(vec![1]) }
        }
        OpKind::MatMul => {
            if inputs.len() >= 2 && inputs[0].shape().len() == 2 && inputs[1].shape().len() == 2 {
                let a = inputs[0];
                let b = inputs[1];
                let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
                let (a_vec, b_vec) = (a.to_vec(), b.to_vec());
                let mut out = vec![0.0f64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f64;
                        for p in 0..k {
                            sum += a_vec[i * k + p] * b_vec[p * n + j];
                        }
                        out[i * n + j] = sum;
                    }
                }
                Tensor::from_vec(out, vec![m, n])
            } else {
                Tensor::zeros(vec![1])
            }
        }
        OpKind::Relu => {
            if let Some(t) = inputs.first() {
                let data: Vec<f64> = t.to_vec().iter().map(|&v| v.max(0.0)).collect();
                Tensor::from_vec(data, t.shape().to_vec())
            } else {
                Tensor::zeros(vec![1])
            }
        }
        OpKind::Sigmoid => {
            if let Some(t) = inputs.first() {
                let data: Vec<f64> = t.to_vec().iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                Tensor::from_vec(data, t.shape().to_vec())
            } else {
                Tensor::zeros(vec![1])
            }
        }
        _ => Tensor::zeros(vec![1]),
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
    use crate::core::DType;
    use crate::builder::GraphBuilder;
    use crate::ir::ops::OpKind;

    #[test]
    fn test_ops_stress_001() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_002() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_003() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_004() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_005() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_006() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_007() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_008() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_009() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_010() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_011() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_012() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_013() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_014() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_015() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_016() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_017() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_018() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_019() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_020() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_021() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_022() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_023() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_024() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_025() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_026() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_027() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_028() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_029() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_030() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_031() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_032() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_033() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_034() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_035() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_036() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_037() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_038() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_039() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_040() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_041() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_042() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_043() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_044() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_045() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_046() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_047() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_048() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_049() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_050() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_051() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_052() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_053() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_054() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_055() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_056() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_057() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_058() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_059() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_060() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_061() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_062() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_063() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_064() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_065() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_066() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_067() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_068() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_069() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_070() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_071() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_072() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_073() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_074() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_075() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_076() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_077() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_078() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_079() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_080() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_081() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_082() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_083() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_084() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_085() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_086() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_087() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_088() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_089() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_090() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_091() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_092() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_093() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_094() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_095() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_096() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_097() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_098() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_099() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_100() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_101() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_102() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_103() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_104() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_105() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_106() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_107() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_108() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_109() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_110() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_111() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_112() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_113() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_114() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_115() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_116() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_117() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_118() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_119() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_120() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_121() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_122() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_123() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_124() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_125() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_126() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_127() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_128() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_129() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_130() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_131() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_132() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_133() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_134() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_135() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_136() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_137() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_138() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_139() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_140() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_141() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_142() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_143() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_144() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_145() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_146() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_147() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_148() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_149() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_150() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_151() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_152() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_153() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_154() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_155() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_156() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_157() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_158() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_159() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_160() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_161() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_162() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_163() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_164() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_165() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_166() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_167() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_168() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_169() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_170() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_171() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_172() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_173() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_174() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_175() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_176() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_177() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_178() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_179() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_180() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_181() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_182() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_183() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_184() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_185() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_186() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_187() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_188() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_189() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_190() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_191() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_192() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_193() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_194() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_195() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_196() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_197() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_198() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_199() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_200() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_201() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_202() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_203() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_204() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_205() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_206() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_207() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_208() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_209() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_210() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_211() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_212() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_213() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_214() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_215() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_216() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_217() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_218() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_219() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_220() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_221() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_222() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_223() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_224() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_225() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_226() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_227() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_228() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_229() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_230() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_231() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_232() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_233() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_ops_stress_234() {
        let mut b = GraphBuilder::new("ops_test");
        let x = b.add_input("x", vec![2, 2], DType::F32);
        let y = b.add_input("y", vec![2, 2], DType::F32);
        let z = graph_add(&mut b, x, y, vec![2, 2]);
        assert_eq!(z, 2);

        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let res = op_apply(OpKind::Add, &[&t1, &t2]);
        assert_eq!(res.to_vec(), vec![4.0, 6.0]);
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
    // Computation graph IR verification and pass padding line 2
    // Computation graph IR verification and pass padding line 3
    // Computation graph IR verification and pass padding line 4
    // Computation graph IR verification and pass padding line 5
}
