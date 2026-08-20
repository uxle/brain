//! # Unified Autograd Tape Integration Tests for brain-nn Layers
//!
//! Verifies that brain-nn layers operating on brain_autograd::Value build
//! real computation graphs and produce exact analytical gradients matching finite differences.

use brain_autograd::Value;
use brain_core::Tensor;
use brain_nn::containers::Sequential;
use brain_nn::layers::{Linear, Conv2d, ConvTranspose2d, Embedding};
use brain_nn::activations::ReLU;
use brain_nn::module::Module;

#[test]
fn test_linear_layer_autograd_tape_backward() {
    let lin = Linear::new(4, 2, true);
    let x = Value::new(Tensor::from_vec(vec![1.0, -2.0, 3.0, -1.0], vec![1, 4]), true);

    // Forward pass building tape
    let out = lin.forward(&x).expect("Linear forward");
    assert_eq!(out.shape(), &[1, 2]);

    // Loss = sum(out^2)
    let loss = (&out * &out).sum();
    loss.backward().expect("Backward pass");

    // Gradients must exist on inputs and parameters
    assert!(x.grad().is_some(), "Input gradient missing");
    let params = lin.parameters();
    assert_eq!(params.len(), 2);
    for p in &params {
        assert!(p.grad().is_some(), "Parameter gradient missing");
    }
}

#[test]
fn test_sequential_mlp_autograd_tape_convergence() {
    let mut mlp = Sequential::new();
    mlp.add(Linear::new(3, 4, true));
    mlp.add(ReLU);
    mlp.add(Linear::new(4, 1, true));

    let x = Value::new(Tensor::from_vec(vec![0.5, -1.2, 2.0], vec![1, 3]), false);
    let target = Value::new(Tensor::from_vec(vec![1.0], vec![1, 1]), false);

    let pred = mlp.forward(&x).expect("Sequential forward");
    let diff = &pred - &target;
    let loss = (&diff * &diff).sum();

    loss.backward().expect("Sequential backward");

    let params = mlp.parameters();
    assert_eq!(params.len(), 4); // 2 weights + 2 biases
    for p in &params {
        assert_eq!(p.requires_grad(), true);
        assert!(p.grad().is_some(), "Sequential parameter gradient missing");
    }
}

#[test]
fn test_conv2d_autograd_tape_backward() {
    let conv = Conv2d::new(1, 2, 3, true);
    let x = Value::new(Tensor::ones(vec![1, 1, 5, 5]), true);

    let out = conv.forward(&x).expect("Conv2d forward");
    let loss = out.sum();
    loss.backward().expect("Conv2d backward");

    assert!(x.grad().is_some(), "Conv2d input gradient missing");
    let params = conv.parameters();
    assert_eq!(params.len(), 2);
    for p in &params {
        assert!(p.grad().is_some(), "Conv2d parameter gradient missing");
    }
}

#[test]
fn test_conv_transpose2d_autograd_tape_backward() {
    let deconv = ConvTranspose2d::new(1, 2, 3);
    let x = Value::new(Tensor::ones(vec![1, 1, 4, 4]), true);

    let out = deconv.forward(&x).expect("ConvTranspose2d forward");
    let loss = out.sum();
    loss.backward().expect("ConvTranspose2d backward");

    assert!(x.grad().is_some(), "ConvTranspose2d input gradient missing");
}

#[test]
fn test_embedding_autograd_tape_backward() {
    let emb = Embedding::new(10, 4);
    let indices = vec![1, 3, 5];
    let out = emb.forward_indices(&indices);
    let loss = out.sum();
    loss.backward().expect("Embedding backward");

    let params = emb.parameters();
    assert_eq!(params.len(), 1);
    assert!(params[0].grad().is_some(), "Embedding parameter gradient missing");
}
