//! # Brain Compile Equivalence & Optimization Verification Tests

use brain_compile::backend::interp::Interpreter;
use brain_compile::ir::{IrGraph, IrType, OpKind};
use brain_core::Tensor;

#[test]
fn test_compile_graph_eval_equivalence() {
    let mut graph = IrGraph::new();

    // Inputs: x [1, 2] and y [1, 2]
    let x_id = graph.add_value(IrType::F64, vec![1, 2]);
    let y_id = graph.add_value(IrType::F64, vec![1, 2]);
    let out_id = graph.add_value(IrType::F64, vec![1, 2]);

    graph.inputs = vec![x_id, y_id];
    graph.outputs = vec![out_id];

    // Node: Add(x, y) -> out
    graph.add_node(OpKind::Add, vec![x_id, y_id], out_id);

    let interp = Interpreter::new();
    let x = Tensor::from_vec(vec![1.5, -2.5], vec![1, 2]);
    let y = Tensor::from_vec(vec![0.5, 3.5], vec![1, 2]);

    let res = interp.evaluate(&graph, &[x.clone(), y.clone()]).unwrap();
    assert_eq!(res.len(), 1);

    let out_tensor = &res[0];
    let expected = &x + &y;

    assert_eq!(out_tensor.shape(), expected.shape());
    for (a, b) in out_tensor.data().iter().zip(expected.data()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Compiled output diff: got {}, expected {}",
            a,
            b
        );
    }
}

#[test]
fn test_constant_folding_and_dce_pipeline() {
    use brain_compile::core::CompileOptions;
    use brain_compile::passes::PassManager;

    let mut graph = IrGraph::new();

    // Constant values
    let c1 = graph.add_value(IrType::F64, vec![1]);
    let c2 = graph.add_value(IrType::F64, vec![1]);
    let sum = graph.add_value(IrType::F64, vec![1]);
    let dead_node_out = graph.add_value(IrType::F64, vec![1]);
    let final_out = graph.add_value(IrType::F64, vec![1]);

    graph.outputs = vec![final_out];

    graph.add_node(OpKind::Constant(5.0), vec![], c1);
    graph.add_node(OpKind::Constant(10.0), vec![], c2);
    graph.add_node(OpKind::Add, vec![c1, c2], sum);
    graph.add_node(OpKind::Relu, vec![sum], final_out);

    // Unconnected dead node
    graph.add_node(OpKind::Sin, vec![c1], dead_node_out);

    assert_eq!(graph.nodes.len(), 5);

    let opts = CompileOptions::default();
    let pm = PassManager::from_options(&opts);
    let changed = pm.run(&mut graph).unwrap();
    assert!(changed);

    // Dead node should be pruned by DCE
    assert!(!graph.nodes.iter().any(|n| n.output == dead_node_out));

    // Evaluate optimized graph
    let interp = Interpreter::new();
    let res = interp.evaluate(&graph, &[]).unwrap();
    assert_eq!(res.len(), 1);
    // Relu(5 + 10) = 15.0
    assert_eq!(res[0].data()[0], 15.0);
}
