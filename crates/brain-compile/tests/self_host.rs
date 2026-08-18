//! # Self-Hosting Compiler Verification Tests

use brain_compile::backend::interp::Interpreter;
use brain_compile::ir::{IrGraph, IrType, OpKind};
use brain_core::Tensor;

#[test]
fn test_self_hosting_ir_interpretation() {
    // Stage 1: Build graph for f(x) = (x + x) * x
    let mut g1 = IrGraph::new();
    let x = g1.add_value(IrType::F64, vec![2]);
    let double_x = g1.add_value(IrType::F64, vec![2]);
    let out = g1.add_value(IrType::F64, vec![2]);

    g1.inputs = vec![x];
    g1.outputs = vec![out];
    g1.add_node(OpKind::Add, vec![x, x], double_x);
    g1.add_node(OpKind::Mul, vec![double_x, x], out);

    // Stage 2: Evaluate with interpreter
    let interp = Interpreter::new();
    let input = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
    let res = interp.evaluate(&g1, &[input.clone()]).unwrap();

    assert_eq!(res.len(), 1);
    let y = &res[0];

    // Reference: (3+3)*3 = 18, (4+4)*4 = 32
    assert_eq!(y.data(), &[18.0, 32.0]);
}
