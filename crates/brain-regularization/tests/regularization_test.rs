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
    assert!(
        (train_mean - 1.0).abs() < 0.15,
        "Inverted dropout mean should preserve expectation: got {}",
        train_mean
    );

    // Eval mode: exact identity
    dropout.eval_mode();
    let eval_out = dropout.apply(&x).expect("Dropout eval");
    assert_eq!(eval_out.to_vec(), x.to_vec(), "Eval mode must be identity");
}

#[test]
fn test_ewc_continual_learning_penalty() {
    use brain_regularization::ewc::ElasticWeightConsolidation;

    let mut ewc = ElasticWeightConsolidation::new(100.0);
    let p1 = Tensor::from_slice(&[1.0, 2.0], vec![2]);
    let grads = vec![vec![0.5, 0.5]];

    ewc.register_task(&[p1.clone()], &grads);

    // Identical parameters -> penalty should be 0
    let penalty0 = ewc.compute_penalty(&[p1]);
    assert!((penalty0 - 0.0).abs() < 1e-6);

    // Shifted parameters -> penalty > 0
    let p2 = Tensor::from_slice(&[1.1, 1.9], vec![2]);
    let penalty1 = ewc.compute_penalty(&[p2]);
    assert!(penalty1 > 0.0);
}

#[test]
fn test_alpha_dropout_and_ewc_gradients() {
    use brain_regularization::dropout::AlphaDropout;
    use brain_regularization::ewc::ElasticWeightConsolidation;

    // Alpha Dropout
    let mut alpha_drop = AlphaDropout::new(0.2);
    let x = Tensor::from_slice(&[0.0; 1000], vec![1000]);
    let out = alpha_drop.apply(&x).unwrap();
    assert_eq!(out.shape(), &[1000]);

    // EWC Gradients
    let mut ewc = ElasticWeightConsolidation::new(50.0);
    let p = Tensor::from_slice(&[2.0, 3.0], vec![2]);
    let grads = vec![vec![1.0, 1.0]];
    ewc.register_task(&[p.clone()], &grads);

    let p_new = Tensor::from_slice(&[2.5, 2.5], vec![2]);
    let penalty_grads = ewc.compute_gradients(&[p_new]);
    assert_eq!(penalty_grads.len(), 1);
    assert_eq!(penalty_grads[0].shape(), &[2]);
    // Grad for p[0]: lambda * (1 + 1e-4) * (2.5 - 2.0) = 50.0 * 1.0001 * 0.5 ~ 25.0
    assert!((penalty_grads[0].data()[0] - 25.0).abs() < 0.1);
}
