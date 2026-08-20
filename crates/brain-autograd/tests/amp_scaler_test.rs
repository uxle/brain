//! # Automatic Mixed Precision (AMP) GradScaler Verification Tests

use brain_autograd::engine::mixed::GradScaler;
use brain_autograd::Value;
use brain_core::Tensor;

#[test]
fn test_grad_scaler_scale_and_unscale() {
    let mut scaler = GradScaler::new(65536.0, 2.0, 0.5, 100);

    let x = Value::new(Tensor::scalar(3.0), true);
    // f(x) = x^2, df/dx = 2*x = 6.0
    let loss = &x * &x;

    let scaled_loss = scaler.scale_loss(&loss);
    scaled_loss.backward().unwrap();

    // Prior to unscaling, grad should be 6.0 * 65536.0
    let raw_grad = x.grad().unwrap().get(0);
    assert!(
        (raw_grad - 6.0 * 65536.0).abs() < 1e-3,
        "Expected scaled grad {}, got {}",
        6.0 * 65536.0,
        raw_grad
    );

    // Unscale gradients: returns Ok(true) if clean, Ok(false) if inf/nan detected
    let success = scaler.unscale_grads(&[&x]).unwrap();
    assert!(success, "Should not have overflowed");

    let unscaled_grad = x.grad().unwrap().get(0);
    assert!(
        (unscaled_grad - 6.0).abs() < 1e-5,
        "Expected unscaled grad 6.0, got {}",
        unscaled_grad
    );
}

#[test]
fn test_grad_scaler_overflow_backoff() {
    let mut scaler = GradScaler::new(1024.0, 2.0, 0.5, 100);
    let x = Value::new(Tensor::scalar(1.0), true);

    // Set an infinite gradient on x
    x.accumulate_grad(&Tensor::scalar(f64::INFINITY)).unwrap();

    let success = scaler.unscale_grads(&[&x]).unwrap();
    assert!(!success, "Should detect infinity");

    scaler.update();
    assert_eq!(
        scaler.scale_factor(),
        512.0,
        "Scale factor should back off by 0.5"
    );
}
