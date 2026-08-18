//! # Tape & Deep Graph Bounded Memory Verification

use brain_core::Tensor;
use brain_autograd::Value;
use brain_autograd::tape::{Tape, OpRecord};

#[test]
fn test_repeated_forward_backward_memory_bounded() {
    let mut tape = Tape::new();
    for _ in 0..100 {
        let x = Value::new(Tensor::scalar(2.0), true);
        let y = &x * &x;
        y.backward().unwrap();
        assert!((x.grad().unwrap().get(0) - 4.0).abs() < 1e-6);

        tape.record(OpRecord::new("mul", vec![1, 2], vec![3], vec![vec![1]]));
        tape.drain();
        assert_eq!(tape.op_count(), 0);
    }
}

#[test]
fn test_deep_chain_100k_bounded_stack() {
    let base = Value::new(Tensor::scalar(1.5), true);
    let mut v = base.clone();

    // Chain 100,000 unary operations
    for _ in 0..100_000 {
        v = v.relu();
    }

    v.backward().unwrap();
    let grad_val = base.grad().unwrap().get(0);
    assert!((grad_val - 1.0).abs() < 1e-6, "Expected grad=1.0 through 100k relu chain, got {}", grad_val);

    // Drop deep graph and ensure zero stack overflow
    drop(v);
}
