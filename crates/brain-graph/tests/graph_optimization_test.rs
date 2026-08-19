use brain_core::Tensor;
use brain_graph::builder::GraphBuilder;
use brain_graph::core::DType;
use brain_graph::interp::GraphInterpreter;
use brain_graph::ir::ops::OpKind;
use brain_graph::passes::const_fold::fold_constants;
use brain_graph::passes::dead_code::eliminate_dead_code;

#[test]
fn test_computational_graph_forward_execution() {
    let mut builder = GraphBuilder::new("linear_relu_mlp");

    // Input x: [2, 2]
    let x_val = builder.add_input("x", vec![2, 2], DType::F32);

    // Weights w: [2, 2] = [[1, 0], [0, 1]]
    let w_val = builder.add_constant("w", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);

    // Bias b: [2, 2] = [[0.5, -0.5], [0.5, -0.5]]
    let b_val = builder.add_constant("b", vec![2, 2], vec![0.5, -0.5, 0.5, -0.5]);

    // Matmul: x @ w
    let mm_out = builder.add_node("matmul_1", OpKind::MatMul, vec![x_val, w_val], vec![2, 2]);

    // Add: mm + b
    let add_out = builder.add_node("add_1", OpKind::Add, vec![mm_out, b_val], vec![2, 2]);

    // ReLU: relu(add)
    let relu_out = builder.add_node("relu_1", OpKind::Relu, vec![add_out], vec![2, 2]);

    builder.mark_output(relu_out);

    let graph = builder.build().expect("Graph build");

    let mut interp = GraphInterpreter::new();
    let x_tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let outputs = interp.run(&graph, &[x_tensor]).expect("Interpretation");
    assert_eq!(outputs.len(), 1);

    let out = &outputs[0];
    assert_eq!(out.shape(), &[2, 2]);

    // Expected: x @ I = x -> x + b = [[1.5, 1.5], [3.5, 3.5]] -> relu = [[1.5, 1.5], [3.5, 3.5]]
    assert_eq!(out.to_vec(), vec![1.5, 1.5, 3.5, 3.5]);
}

#[test]
fn test_dead_code_elimination_pass() {
    let mut builder = GraphBuilder::new("dce_test");

    let x = builder.add_input("x", vec![2, 2], DType::F32);
    let c1 = builder.add_constant("c1", vec![2, 2], vec![1.0; 4]);
    let c2 = builder.add_constant("c2", vec![2, 2], vec![2.0; 4]);

    // Used path
    let used_add = builder.add_node("used_add", OpKind::Add, vec![x, c1], vec![2, 2]);
    builder.mark_output(used_add);

    // Dead path (never used by outputs)
    let _dead_mul = builder.add_node("dead_mul", OpKind::Mul, vec![x, c2], vec![2, 2]);

    let mut graph = builder.build().expect("Graph build");
    assert_eq!(graph.nodes.len(), 2);

    eliminate_dead_code(&mut graph).expect("DCE pass");
    assert_eq!(graph.nodes.len(), 1, "DCE must eliminate the unused node");
    assert_eq!(graph.nodes[0].name, "used_add");
}

#[test]
fn test_constant_folding_pass() {
    let mut builder = GraphBuilder::new("const_fold_test");

    let c1 = builder.add_constant("c1", vec![2, 2], vec![2.0; 4]);
    let c2 = builder.add_constant("c2", vec![2, 2], vec![3.0; 4]);

    // Foldable constant operation: c1 * c2
    let const_mul = builder.add_node("const_mul", OpKind::Mul, vec![c1, c2], vec![2, 2]);

    // Dynamic input
    let x = builder.add_input("x", vec![2, 2], DType::F32);
    let final_add = builder.add_node("final_add", OpKind::Add, vec![x, const_mul], vec![2, 2]);
    builder.mark_output(final_add);

    let mut graph = builder.build().expect("Graph build");
    assert_eq!(graph.nodes.len(), 2);

    let modified = fold_constants(&mut graph).expect("Const fold pass");
    assert!(modified);
    assert_eq!(graph.nodes[0].op, OpKind::Constant, "Constant node must be folded");
}
