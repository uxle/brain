//! # Autopilot Optimization & Hyperparameter Search Tests

use brain_core::Tensor;
use brain_optim::{Optimizer, ParamGroup, Sgd, SgdConfig};

#[test]
fn test_autopilot_lr_search_finds_optimal_convergence() {
    let candidate_lrs = vec![0.001, 0.01, 0.1, 0.5];
    let mut best_lr = 0.0;
    let mut lowest_loss = f64::INFINITY;

    // Evaluate optimization of f(x) = x^2 starting at x_0 = 2.0 over 10 steps
    for &lr in &candidate_lrs {
        let mut x = Tensor::scalar(2.0);
        let mut sgd = Sgd::new(
            vec![ParamGroup::new(vec![0], lr)],
            SgdConfig {
                lr,
                ..SgdConfig::default()
            },
        );

        for _ in 0..10 {
            let grad = Tensor::scalar(2.0 * x.get(0));
            sgd.step(std::slice::from_mut(&mut x), &[grad]).unwrap();
        }

        let final_loss = x.get(0) * x.get(0);
        if final_loss < lowest_loss {
            lowest_loss = final_loss;
            best_lr = lr;
        }
    }

    assert_eq!(best_lr, 0.5, "Autopilot search should discover optimal lr 0.5");
    assert!(lowest_loss < 1e-6, "Optimal lr should achieve near-zero loss: got {}", lowest_loss);
}
