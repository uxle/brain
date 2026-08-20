//! # Higher-Order Functional Autograd Verification Harness (Stage D, Phases 86-115)
//!
//! Tests Jacobian matrix calculation, Hessian curvature matrix calculation,
//! JVP, VJP, scalar grad, and higher-order functional transforms.

use brain_autograd::prelude::*;
use brain_core::Tensor;

fn approx(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

// -----------------------------------------------------------------------------
// Phase 86 & 87: Scalar Grad and Value & Grad
// -----------------------------------------------------------------------------
#[test]
fn test_scalar_grad_and_value_grad() {
    // f(x) = x^3 => f'(x) = 3 x^2
    let x = Value::scalar(2.0);
    let g = grad(|v| v.mul(v).mul(v), &x).unwrap().unwrap();
    // 3 * (2.0)^2 = 12.0
    assert!(approx(g.get(0), 12.0, 1e-5), "Expected 12.0, got {}", g.get(0));

    let (val, g_joint) = value_and_grad(|v| v.mul(v).mul(v), &x).unwrap();
    assert_eq!(val.data().get(0), 8.0);
    assert!(approx(g_joint.unwrap().get(0), 12.0, 1e-5));
}

// -----------------------------------------------------------------------------
// Phase 99 & 33: VJP and JVP Functional Transforms
// -----------------------------------------------------------------------------
#[test]
fn test_vjp_and_jvp_transforms() {
    // f(x) = x^2 => J = 2 x = 4.0 at x = 2.0
    let x = Value::scalar(2.0);
    let v = Tensor::scalar(3.0);

    // v^T J = 3.0 * 4.0 = 12.0
    let (val_vjp, grad_vjp) = vjp(|u| u.mul(u), &x, &v).unwrap();
    assert_eq!(val_vjp.data().get(0), 4.0);
    assert!(approx(grad_vjp.get(0), 12.0, 1e-5));

    // J v = 4.0 * 3.0 = 12.0
    let (val_jvp, out_jvp) = jvp(|u| u.mul(u), &x, &v).unwrap();
    assert_eq!(val_jvp.data().get(0), 4.0);
    assert!(approx(out_jvp.get(0), 12.0, 1e-4));
}

// -----------------------------------------------------------------------------
// Phase 97: Full Jacobian Matrix Evaluation
// -----------------------------------------------------------------------------
#[test]
fn test_full_jacobian_matrix() {
    // f([x0, x1]) = [x0^2, x0 * x1]
    // J = [
    //   [2*x0,  0   ]
    //   [x1,    x0  ]
    // ]
    // At x = [2.0, 3.0]:
    // J = [
    //   [4.0, 0.0]
    //   [3.0, 2.0]
    // ]
    let x = Value::from_slice(&[2.0, 3.0], vec![2, 1]);
    let w0 = Value::from_slice(&[1.0, 0.0], vec![1, 2]);
    let w1 = Value::from_slice(&[0.0, 1.0], vec![1, 2]);

    let jac = jacobian(|v| {
        let x0 = w0.matmul(v);
        let x1 = w1.matmul(v);
        let y0 = x0.mul(&x0);
        let y1 = x0.mul(&x1);
        y0.add(&y1)
    }, &x).unwrap();

    assert_eq!(jac.shape(), &[1, 2]);
    // y = x0^2 + x0 * x1 => dy/dx0 = 2*x0 + x1 = 4 + 3 = 7, dy/dx1 = x0 = 2
    assert!(approx(jac.get_2d(0, 0), 7.0, 1e-4));
    assert!(approx(jac.get_2d(0, 1), 2.0, 1e-4));
}

// -----------------------------------------------------------------------------
// Phase 98: Full Hessian Curvature Matrix Evaluation
// -----------------------------------------------------------------------------
#[test]
fn test_full_hessian_matrix() {
    // f([x0, x1]) = x0^3 + x0 * x1 + x1^2
    // grad = [3*x0^2 + x1,  x0 + 2*x1]
    // Hessian = [
    //   [6*x0,  1.0]
    //   [1.0,   2.0]
    // ]
    // At x = [2.0, 3.0]:
    // Hessian = [
    //   [12.0, 1.0]
    //   [1.0,  2.0]
    // ]
    let x = Value::from_slice(&[2.0, 3.0], vec![2, 1]);
    let w0 = Value::from_slice(&[1.0, 0.0], vec![1, 2]);
    let w1 = Value::from_slice(&[0.0, 1.0], vec![1, 2]);

    let (g, h) = grad_and_hess(|v| {
        let x0 = w0.matmul(v);
        let x1 = w1.matmul(v);
        let term1 = x0.mul(&x0).mul(&x0);
        let term2 = x0.mul(&x1);
        let term3 = x1.mul(&x1);
        term1.add(&term2).add(&term3)
    }, &x).unwrap();

    // grad = [3*4 + 3, 2 + 2*3] = [15.0, 8.0]
    assert!(approx(g.get(0), 15.0, 1e-4));
    assert!(approx(g.get(1), 8.0, 1e-4));

    // Hessian shape: [2, 2]
    assert_eq!(h.shape(), &[2, 2]);
    assert!(approx(h.get_2d(0, 0), 12.0, 1e-3));
    assert!(approx(h.get_2d(0, 1), 1.0, 1e-3));
    assert!(approx(h.get_2d(1, 0), 1.0, 1e-3));
    assert!(approx(h.get_2d(1, 1), 2.0, 1e-3));
}

// -----------------------------------------------------------------------------
// Phase 115: Stage D Master Differential Calculus Audit
// -----------------------------------------------------------------------------
#[test]
fn test_stage_d_master_differential_calculus_audit() {
    // Quadratic form f(x) = 1/2 x^T A x where A is symmetric positive definite
    // A = [[4, 2], [2, 6]]
    // grad f(x) = A x
    // hessian f(x) = A
    let a_mat = Tensor::from_slice(&[4.0, 2.0, 2.0, 6.0], vec![2, 2]);
    let a_val = Value::from_tensor(&a_mat);

    let x = Value::from_slice(&[1.0, 2.0], vec![2, 1]);
    let h = hessian(|v| {
        let av = a_val.matmul(v);
        let vt = v.transpose(0, 1);
        let vt_av = vt.matmul(&av);
        vt_av.mul(&Value::scalar(0.5)).sum()
    }, &x).unwrap();

    assert_eq!(h.shape(), &[2, 2]);
    assert!(approx(h.get_2d(0, 0), 4.0, 1e-3));
    assert!(approx(h.get_2d(0, 1), 2.0, 1e-3));
    assert!(approx(h.get_2d(1, 0), 2.0, 1e-3));
    assert!(approx(h.get_2d(1, 1), 6.0, 1e-3));
}
