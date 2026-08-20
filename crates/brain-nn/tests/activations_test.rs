//! # Activations Verification Test Suite

use brain_core::Tensor;
use brain_nn::{celu, elu, glu, hard_sigmoid, hard_swish, selu, softplus, softsign, swiglu};

#[test]
fn test_elu_and_celu() {
    let t = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let out_elu = elu(&t, 1.0);
    assert!((out_elu.get(3) - 1.0).abs() < 1e-6);
    assert!((out_elu.get(4) - 2.0).abs() < 1e-6);
    assert!((out_elu.get(2) - 0.0).abs() < 1e-6);
    assert!(out_elu.get(0) < 0.0 && out_elu.get(0) > -1.0);

    let out_celu = celu(&t, 1.0);
    assert_eq!(out_celu.shape(), &[5]);
}

#[test]
fn test_selu_and_softplus() {
    let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]);
    let out_selu = selu(&t);
    assert!(out_selu.get(2) > 1.0); // 1.0507 * 1.0

    let out_softplus = softplus(&t, 1.0);
    assert!(out_softplus.get(0) > 0.0);
    assert!(out_softplus.get(1) > 0.0);
    assert!(out_softplus.get(2) > 1.0);
}

#[test]
fn test_hard_activations() {
    let t = Tensor::from_vec(vec![-4.0, 0.0, 4.0], vec![3]);
    let hsig = hard_sigmoid(&t);
    assert_eq!(hsig.get(0), 0.0);
    assert_eq!(hsig.get(1), 0.5);
    assert_eq!(hsig.get(2), 1.0);

    let hswish = hard_swish(&t);
    assert_eq!(hswish.get(0), 0.0);
    assert_eq!(hswish.get(1), 0.0);
    assert_eq!(hswish.get(2), 4.0);

    let ssign = softsign(&t);
    assert!((ssign.get(0) - (-4.0 / 5.0)).abs() < 1e-6);
}

#[test]
fn test_glu_and_swiglu() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let out_glu = glu(&t, 1);
    assert_eq!(out_glu.shape(), &[1, 2]);

    let out_swiglu = swiglu(&t, 1);
    assert_eq!(out_swiglu.shape(), &[1, 2]);
}
