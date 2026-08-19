//! # Regularization Integration Tests

use brain_core::Tensor;
use brain_regularization::core::Regularization;
use brain_regularization::dropout::Dropout;

#[test]
fn test_inverted_dropout_scaling_and_eval_mode() {
    let mut dropout = Dropout::with_seed(0.5, 42);

    let x = Tensor::ones(vec![1000]);

    // Training mode: expected mean ≈ 1.0 (inverted scaling 1 / 0.5 = 2.0 on survivors)
    let train_out = dropout.apply(&x).expect("Dropout train");
    let train_sum: f64 = train_out.data().iter().sum();
    let train_mean = train_sum / 1000.0;
    assert!((train_mean - 1.0).abs() < 0.15, "Inverted dropout mean should preserve expectation: got {}", train_mean);

    // Eval mode: exact identity
    dropout.eval_mode();
    let eval_out = dropout.apply(&x).expect("Dropout eval");
    assert_eq!(eval_out.to_vec(), x.to_vec(), "Eval mode must be identity");
}
