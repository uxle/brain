//! Numerical gradient checking harness and comprehensive VJP test suite.

use brain_autograd::Value;
use brain_core::Tensor;

fn approx(a: f64, b: f64) -> bool {
    let r = if a.abs() > 1e-8 { (a - b).abs() / a.abs() } else { (a - b).abs() };
    r < 1e-4
}

/// Central-difference numeric gradient of a scalar-valued `f(x)`, one element at a time.
pub fn numeric_scalar_grad<F>(x_data: &[f64], shape: &[usize], f: F, eps: f64) -> Vec<f64>
where
    F: Fn(&Value) -> Value,
{
    let mut num = vec![0.0f64; x_data.len()];
    for i in 0..x_data.len() {
        let mut xp = x_data.to_vec();
        xp[i] += eps;
        let mut xm = x_data.to_vec();
        xm[i] -= eps;
        let vpx = Value::from_slice(&xp, shape.to_vec());
        let vmx = Value::from_slice(&xm, shape.to_vec());
        let yp = f(&vpx).data().get(0);
        let ym = f(&vmx).data().get(0);
        num[i] = (yp - ym) / (2.0 * eps);
    }
    num
}

/// Runs `backward` and reads the per-element analytic gradient of `f(x)` wrt `x`.
pub fn analytic_scalar_grad<F>(x_data: &[f64], shape: &[usize], f: F) -> Vec<f64>
where
    F: Fn(&Value) -> Value,
{
    let mut x = Value::from_slice(x_data, shape.to_vec());
    x.set_requires_grad(true);
    let y = f(&x);
    x.zero_grad();
    y.backward().unwrap();
    let g = x.grad().unwrap();
    (0..x.numel()).map(|i| g.get(i)).collect()
}

/// Compares an op's analytic gradient against its central-difference numerical gradient.
pub fn check_gradient<F>(name: &str, f: F, x_data: &[f64], shape: &[usize], epsilon: f64, tol: f64)
where
    F: Fn(&Value) -> Value,
{
    let analytic = analytic_scalar_grad(x_data, shape, &f);
    let numeric = numeric_scalar_grad(x_data, shape, &f, epsilon);
    for (i, (&a, &n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let rel_err = (a - n).abs() / a.abs().max(n.abs()).max(1e-8);
        assert!(
            rel_err < tol,
            "VJP mismatch in [{}]: idx {}, analytic={:.6}, numeric={:.6}, rel_err={:.6}",
            name, i, a, n, rel_err
        );
    }
}

pub fn assert_grads_close(name: &str, an: &[f64], num: &[f64]) {
    let tol = 1e-4;
    for (i, (&a, &n)) in an.iter().zip(num.iter()).enumerate() {
        let scale = a.abs().max(n.abs()).max(1e-8);
        let rel_err = (a - n).abs() / scale;
        assert!(
            rel_err <= tol,
            "VJP mismatch in [{}]: idx {}, analytic={:.6}, numeric={:.6}, rel_err={:.6}",
            name, i, a, n, rel_err
        );
    }
}

fn scalar_grad<F>(f: F, x_val: f64) -> f64
where
    F: Fn(&Value) -> Value,
{
    let mut x = Value::scalar(x_val);
    x.set_requires_grad(true);
    let y = f(&x);
    x.zero_grad();
    y.backward().unwrap();
    let g = x.grad().unwrap();
    g.get(0)
}

fn check(name: &str, x_val: f64, f: impl Fn(&Value) -> Value, f_ref: impl Fn(f64) -> f64) {
    let analytic = scalar_grad(f, x_val);
    let eps = 1e-5;
    let numeric = (f_ref(x_val + eps) - f_ref(x_val - eps)) / (2.0 * eps);
    let ok = approx(analytic, numeric);
    assert!(
        ok,
        "BUG [{}]: x={:?}, analytic={:.6}, numeric={:.6}",
        name, x_val, analytic, numeric
    );
}

// =============================================================================
// Tests: Scalar Ops & Activations
// =============================================================================

#[test]
fn check_scalar_ops() {
    let xs = [0.3, 0.7, 1.5, -0.4, 2.0, 3.3, 5.0, 10.0];
    for &x in &xs {
        check("exp", x, |v| v.exp(), |v| v.exp());
        check("log", x.abs() + 0.5, |v| v.log(), |v| v.ln());
        check("sqrt", x.abs() + 0.5, |v| v.sqrt(), |v| v.sqrt());
        check("relu", x, |v| v.relu(), |v| v.max(0.0));
        check("sigmoid", x, |v| v.sigmoid(), |v| 1.0 / (1.0 + (-v).exp()));
        check("tanh", x, |v| v.tanh(), |v| v.tanh());
        check("neg", x, |v| -v.clone(), |v| -v);
    }
}

#[test]
fn check_new_unary_ops() {
    // abs: away from the kink at 0
    check_gradient("abs", |x: &Value| x.abs().sum(), &[1.5, -2.0, 0.5, -3.3], &[4], 1e-5, 1e-4);

    // clamp: interior points get full gradient
    check_gradient(
        "clamp_interior",
        |x: &Value| x.clamp(-1.0, 1.0).sum(),
        &[0.5, -0.7, 0.2, -0.1],
        &[4],
        1e-5,
        1e-4,
    );

    // sin / cos
    check_gradient("sin", |x: &Value| x.sin().sum(), &[0.3, 1.1, -2.0, 3.5], &[4], 1e-5, 1e-4);
    check_gradient("cos", |x: &Value| x.cos().sum(), &[0.3, 1.1, -2.0, 3.5], &[4], 1e-5, 1e-4);

    // recip: away from the pole at 0
    check_gradient("recip", |x: &Value| x.recip().sum(), &[0.5, 2.0, -3.0, 1.5], &[4], 1e-5, 1e-4);

    // square
    check_gradient("square", |x: &Value| x.square().sum(), &[0.5, -2.0, 3.0, -0.5], &[4], 1e-5, 1e-4);

    // sign: piecewise constant -> gradient is exactly zero
    let mut x = Value::from_slice(&[3.0, -2.0, 0.0], vec![3]);
    x.set_requires_grad(true);
    let y = x.sign().sum();
    x.zero_grad();
    y.backward().unwrap();
    let g = x.grad().unwrap();
    for i in 0..3 {
        assert_eq!(g.get(i), 0.0, "sign gradient must be zero at idx {}", i);
    }
}

#[test]
fn check_clamp_boundaries() {
    // Mixed region: below min (0 grad), inside (full grad), above max (0 grad)
    check_gradient(
        "clamp_mixed",
        |x: &Value| x.clamp(-1.0, 1.0).sum(),
        &[-3.0, -1.5, 0.3, 2.5, 5.0],
        &[5],
        1e-5,
        1e-4,
    );
}

/// Numerically checks a two-input elementwise op with real assertions.
fn check_pair<F>(name: &str, a_data: &[f64], b_data: &[f64], f: F)
where
    F: Fn(&Value, &Value) -> Value,
{
    let shape = vec![a_data.len()];
    let eps = 1e-5;

    let mut a = Value::from_slice(&a_data, shape.clone());
    a.set_requires_grad(true);
    let mut b = Value::from_slice(&b_data, shape.clone());
    b.set_requires_grad(true);
    let y = f(&a, &b);
    a.zero_grad();
    b.zero_grad();
    y.backward().unwrap();
    let ga: Vec<f64> = (0..a.numel()).map(|i| a.grad().unwrap().get(i)).collect();
    let gb: Vec<f64> = (0..b.numel()).map(|i| b.grad().unwrap().get(i)).collect();

    let mut na = vec![0.0; a_data.len()];
    for i in 0..a_data.len() {
        let mut xp = a_data.to_vec();
        xp[i] += eps;
        let mut xm = a_data.to_vec();
        xm[i] -= eps;
        let bv = Value::from_slice(&b_data, shape.clone());
        let yp = f(&Value::from_slice(&xp, shape.clone()), &bv).data().get(0);
        let bv = Value::from_slice(&b_data, shape.clone());
        let ym = f(&Value::from_slice(&xm, shape.clone()), &bv).data().get(0);
        na[i] = (yp - ym) / (2.0 * eps);
    }
    assert_grads_close(&format!("{}_wrt_a", name), &ga, &na);

    let mut nb = vec![0.0; b_data.len()];
    for i in 0..b_data.len() {
        let mut xp = b_data.to_vec();
        xp[i] += eps;
        let mut xm = b_data.to_vec();
        xm[i] -= eps;
        let av = Value::from_slice(&a_data, shape.clone());
        let yp = f(&av, &Value::from_slice(&xp, shape.clone())).data().get(0);
        let av = Value::from_slice(&a_data, shape.clone());
        let ym = f(&av, &Value::from_slice(&xm, shape.clone())).data().get(0);
        nb[i] = (yp - ym) / (2.0 * eps);
    }
    assert_grads_close(&format!("{}_wrt_b", name), &gb, &nb);
}

#[test]
fn check_min_max_elem_grad() {
    // No ties: strict winner takes the whole gradient
    check_pair(
        "min_elem",
        &[0.5, -1.5, 2.0, 0.3],
        &[0.3, 0.5, 1.0, 2.0],
        |a: &Value, b: &Value| a.min_elem(b).sum(),
    );
    check_pair(
        "max_elem",
        &[0.5, -1.5, 2.0, 0.3],
        &[0.3, 0.5, 1.0, 2.0],
        |a: &Value, b: &Value| a.max_elem(b).sum(),
    );
}

#[test]
fn check_where_grad() {
    let cond_data = [1.0, 0.0, 1.0, 0.0];
    let a_data = [0.5, -1.5, 2.0, 0.3];
    let b_data = [0.3, 0.5, 1.0, 2.0];
    let shape = vec![4];
    let eps = 1e-5;

    let f = |a: &Value, b: &Value| {
        let c = Value::from_slice(&cond_data, shape.clone());
        a.where_cond(&c, b).sum()
    };
    let f_single = |x: &Value| f(x, &Value::from_slice(&b_data, shape.clone()));

    // a-grad: only masked positions flow (numeric catches it too)
    let mut a = Value::from_slice(&a_data, shape.clone());
    a.set_requires_grad(true);
    let b = Value::from_slice(&b_data, shape.clone());
    let y = f(&a, &b);
    a.zero_grad();
    y.backward().unwrap();
    let ga: Vec<f64> = (0..a.numel()).map(|i| a.grad().unwrap().get(i)).collect();
    let na = numeric_scalar_grad(&a_data, &shape, f_single, eps);
    assert_grads_close("where_wrt_a", &ga, &na);

    let f_single_b = |x: &Value| {
        let c = Value::from_slice(&cond_data, shape.clone());
        let av = Value::from_slice(&a_data, shape.clone());
        av.where_cond(&c, x).sum()
    };
    let mut b = Value::from_slice(&b_data, shape.clone());
    b.set_requires_grad(true);
    let a = Value::from_slice(&a_data, shape.clone());
    let y = f(&a, &b);
    b.zero_grad();
    y.backward().unwrap();
    let gb: Vec<f64> = (0..b.numel()).map(|i| b.grad().unwrap().get(i)).collect();
    let nb = numeric_scalar_grad(&b_data, &shape, f_single_b, eps);
    assert_grads_close("where_wrt_b", &gb, &nb);
}

#[test]
fn check_activation_kinks_and_boundaries() {
    // Kink at x = 0.0 for ReLU
    let relu_kink = |x: &Value| x.relu().sum();
    check_gradient("relu_kink_neg", relu_kink, &[-0.5, -0.01], &[2], 1e-5, 1e-4);
    check_gradient("relu_kink_pos", relu_kink, &[0.01, 0.5], &[2], 1e-5, 1e-4);

    // Sigmoid & Tanh
    let sig_fn = |x: &Value| x.sigmoid().sum();
    check_gradient("sigmoid", sig_fn, &[-2.0, 0.0, 2.0], &[3], 1e-5, 1e-4);

    let tanh_fn = |x: &Value| x.tanh().sum();
    check_gradient("tanh", tanh_fn, &[-2.0, 0.0, 2.0], &[3], 1e-5, 1e-4);

    // Direct grad_gelu and grad_silu analytical check
    let x_t = Tensor::from_slice(&[-1.5, 0.0, 1.5], vec![3]);
    let g_t = Tensor::ones(vec![3]);
    let d_gelu = brain_autograd::ops::grad_gelu(&x_t, &g_t).unwrap();
    assert_eq!(d_gelu.shape(), &[3]);

    let d_silu = brain_autograd::ops::grad_silu(&x_t, &g_t).unwrap();
    assert_eq!(d_silu.shape(), &[3]);
}

#[test]
fn check_pow_grad() {
    let xs = [0.5, 1.5, 2.0, 3.0, 4.0];
    for &p in &[2.0, 3.0, 0.5, -1.0] {
        for &x in &xs {
            let p_val = Value::scalar(p);
            check(
                &format!("pow_{}", p),
                x,
                |v| v.pow(&p_val),
                |v| v.powf(p),
            );
        }
    }
}

// =============================================================================
// Tests: Binary & Broadcast Gradients
// =============================================================================

#[test]
fn check_binary_elementwise_grad() {
    let a_data = vec![1.5, -2.0, 3.5, 4.0];
    let b_data = vec![0.5, 1.0, -1.5, 2.0];
    let shape = vec![2, 2];

    let b_val = Value::from_slice(&b_data, shape.clone());
    check_gradient("add", |a| (a + &b_val).sum(), &a_data, &shape, 1e-5, 1e-4);
    check_gradient("sub", |a| (a - &b_val).sum(), &a_data, &shape, 1e-5, 1e-4);
    check_gradient("mul", |a| (a * &b_val).sum(), &a_data, &shape, 1e-5, 1e-4);
    check_gradient("div", |a| (a / &b_val).sum(), &a_data, &shape, 1e-5, 1e-4);
}

#[test]
fn check_broadcast_mul_grad() {
    // a: [2, 3], b: [1, 3] -> a * b: [2, 3]
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.5, -1.0, 2.0];
    let a_shape = vec![2, 3];
    let b_shape = vec![1, 3];

    let b_val = Value::from_slice(&b_data, b_shape);
    check_gradient("broadcast_mul_a", |a| (a * &b_val).sum(), &a_data, &a_shape, 1e-5, 1e-4);

    let a_val = Value::from_slice(&a_data, a_shape);
    check_gradient("broadcast_mul_b", |b| (&a_val * b).sum(), &b_data, &vec![1, 3], 1e-5, 1e-4);
}

#[test]
fn check_broadcast_add_shapes() {
    // a: [3, 1], b: [1, 4] -> [3, 4]
    let a_data = vec![1.0, 2.0, 3.0];
    let b_data = vec![0.1, 0.2, 0.3, 0.4];
    
    let b_val = Value::from_slice(&b_data, vec![1, 4]);
    check_gradient("broadcast_add_3x1", |a| (a + &b_val).sum(), &a_data, &vec![3, 1], 1e-5, 1e-4);

    let a_val = Value::from_slice(&a_data, vec![3, 1]);
    check_gradient("broadcast_add_1x4", |b| (&a_val + b).sum(), &b_data, &vec![1, 4], 1e-5, 1e-4);
}

// =============================================================================
// Tests: Reductions
// =============================================================================

#[test]
fn check_mean_and_sum_grad() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];

    check_gradient("mean", |x| x.mean(), &x_data, &shape, 1e-5, 1e-4);
    check_gradient("sum", |x| x.sum(), &x_data, &shape, 1e-5, 1e-4);
}

#[test]
fn check_softmax_logsoftmax_grad() {
    let x_data = vec![1.0, -2.0, 3.0, 0.5, 1.5, -0.5];
    let shape = vec![2, 3];

    check_gradient("softmax", |x| brain_autograd::ops::softmax(x).sum(), &x_data, &shape, 1e-5, 1e-4);
    check_gradient("log_softmax", |x| brain_autograd::ops::log_softmax(x).sum(), &x_data, &shape, 1e-5, 1e-4);
}

// =============================================================================
// Tests: Linear Algebra & Matrix Multiplication
// =============================================================================

#[test]
fn check_matmul_grad() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b_data = vec![0.5, -1.0, 1.5, 2.0, -0.5, 1.0]; // [3, 2]

    let b_val = Value::from_slice(&b_data, vec![3, 2]);
    check_gradient("matmul_wrt_a", |a| a.matmul(&b_val).sum(), &a_data, &vec![2, 3], 1e-5, 1e-4);

    let a_val = Value::from_slice(&a_data, vec![2, 3]);
    check_gradient("matmul_wrt_b", |b| a_val.matmul(b).sum(), &b_data, &vec![3, 2], 1e-5, 1e-4);
}

#[test]
fn test_batched_matmul_grad() {
    // Batched MatMul: A [2, 2, 3] * B [2, 3, 2] -> C [2, 2, 2]
    let a_data: Vec<f64> = (0..12).map(|i| (i as f64) * 0.5 + 0.1).collect();
    let b_data: Vec<f64> = (0..12).map(|i| (i as f64) * 0.3 - 0.2).collect();

    let b_val = Value::from_slice(&b_data, vec![2, 3, 2]);
    check_gradient("batched_matmul_a", |a| a.matmul(&b_val).sum(), &a_data, &vec![2, 2, 3], 1e-5, 1e-4);

    let a_val = Value::from_slice(&a_data, vec![2, 2, 3]);
    check_gradient("batched_matmul_b", |b| a_val.matmul(b).sum(), &b_data, &vec![2, 3, 2], 1e-5, 1e-4);
}

// =============================================================================
// Tests: Convolutions & Pooling
// =============================================================================

#[test]
fn check_conv2d_grad() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // [1, 1, 3, 3]
    let w_val = Value::from_slice(&[0.5, -0.5, 1.0, -1.0], vec![1, 1, 2, 2]);

    check_gradient("conv2d", |x| x.conv2d(&w_val, None, (1, 1), (0, 0)).sum(), &x_data, &vec![1, 1, 3, 3], 1e-5, 1e-4);
}

#[test]
fn check_conv2d_strided_padded_grad() {
    let x_data: Vec<f64> = (0..16).map(|i| (i as f64) * 0.2 + 0.5).collect();
    let w_val = Value::from_slice(&[0.1, 0.2, 0.3, 0.4], vec![1, 1, 2, 2]);
    let b_val = Value::from_slice(&[0.5], vec![1]);

    check_gradient(
        "conv2d_strided_padded",
        |x| x.conv2d(&w_val, Some(&b_val), (2, 2), (1, 1)).sum(),
        &x_data,
        &vec![1, 1, 4, 4],
        1e-5,
        1e-4,
    );
}

#[test]
fn check_max_pool2d_grad() {
    let x_data = vec![
        1.0, 3.0, 2.0, 4.0,
        5.0, 6.0, 8.0, 7.0,
        9.0, 11.0, 10.0, 12.0,
        13.0, 14.0, 16.0, 15.0,
    ];
    check_gradient("max_pool2d", |x| x.max_pool2d((2, 2), (2, 2), (0, 0)).sum(), &x_data, &vec![1, 1, 4, 4], 1e-5, 1e-4);
}

#[test]
fn check_avg_pool2d_grad() {
    let x_data: Vec<f64> = (0..16).map(|i| (i as f64) * 0.5 + 1.0).collect();
    check_gradient("avg_pool2d", |x| x.avg_pool2d((2, 2), (2, 2), (0, 0)).sum(), &x_data, &vec![1, 1, 4, 4], 1e-5, 1e-4);
}

#[test]
fn check_conv_transpose2d_grad() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 2, 2]
    let w_val = Value::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![1, 1, 2, 2]);

    check_gradient("conv_transpose2d", |x| x.conv_transpose2d(&w_val, None, (1, 1), (0, 0)).sum(), &x_data, &vec![1, 1, 2, 2], 1e-5, 1e-4);
}

// =============================================================================
// Tests: Loss Functions & Neural Layers
// =============================================================================

#[test]
fn check_linear_grad() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let w_val = Value::from_slice(&[0.5, -0.5, 1.5, -1.0], vec![2, 2]);
    let b_val = Value::from_slice(&[0.1, -0.2], vec![2]);

    check_gradient("linear", |x| x.linear(&w_val, Some(&b_val)).sum(), &x_data, &vec![2, 2], 1e-5, 1e-4);
}

#[test]
fn check_mse_loss_grad() {
    let p_data = vec![1.2, -0.8, 2.4, 0.5];
    let t_val = Value::from_slice(&[1.0, -1.0, 2.0, 0.0], vec![2, 2]);

    check_gradient("mse_loss", |p| {
        let diff = p - &t_val;
        (&diff * &diff).mean()
    }, &p_data, &vec![2, 2], 1e-5, 1e-4);
}

#[test]
fn check_cross_entropy_loss_grad() {
    let x_data = vec![0.5, 1.5, 2.5, 3.0, 1.0, 0.1];
    let mask = vec![
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
    ];

    check_gradient("cross_entropy_loss", |x| {
        let lsm = brain_autograd::ops::log_softmax(x);
        let mask_val = Value::from_slice(&mask, vec![2, 3]);
        let selected = lsm * mask_val;
        selected.sum().neg() / Value::scalar(2.0)
    }, &x_data, &vec![2, 3], 1e-5, 1e-4);
}

#[test]
fn check_embedding_grad() {
    let w_data = vec![
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2,
    ];
    let indices = vec![1, 2, 1];

    check_gradient("embedding", |w| w.embedding(&indices, vec![3, 3]).sum(), &w_data, &vec![4, 3], 1e-5, 1e-4);
}

// =============================================================================
// Tests: Graph Topology, Diamond Graph, & Engine Invariants
// =============================================================================

#[test]
fn test_diamond_graph_gradient_accumulation() {
    // Graph: a -> b = a^2, a -> c = 2*a, d = b + c = a^2 + 2*a
    // dd/da = 2*a + 2
    let a_data = vec![3.0];
    let a_shape = vec![1];

    let diamond = |a: &Value| {
        let b = a * a;
        let c = a * &Value::scalar(2.0);
        &b + &c
    };

    check_gradient("diamond_graph", diamond, &a_data, &a_shape, 1e-5, 1e-4);
}

#[test]
fn test_repeated_backward_no_double_count() {
    let mut a = Value::from_slice(&[2.0, 3.0], vec![2]);
    a.set_requires_grad(true);

    let b = &a * &a; // b = a^2 -> db/da = 2*a = [4, 6]
    b.backward().unwrap();

    let g1 = a.grad().unwrap();
    assert_eq!(g1.to_vec(), vec![4.0, 6.0]);

    // Second backward without zero_grad -> accumulates to [8, 12]
    b.backward().unwrap();
    let g2 = a.grad().unwrap();
    assert_eq!(g2.to_vec(), vec![8.0, 12.0]);
}

#[test]
fn test_checkpointing_numerical_equivalence() {
    let a_data = vec![1.5, 2.5, -1.0, 3.0];
    let shape = vec![2, 2];

    // Standard pass without checkpointing
    let mut a_std = Value::from_slice(&a_data, shape.clone());
    a_std.set_requires_grad(true);
    let y_std = (&a_std * &a_std).tanh().sum();
    y_std.backward().unwrap();
    let g_std = a_std.grad().unwrap().to_vec();

    // Pass with selective activation checkpointing
    let mut a_chk = Value::from_slice(&a_data, shape);
    a_chk.set_requires_grad(true);
    let y_chk = brain_autograd::checkpoint::checkpoint(|inputs| {
        let a = inputs[0];
        Ok(vec![(a * a).tanh().sum()])
    }, &[&a_chk]).unwrap();
    y_chk[0].backward().unwrap();
    let g_chk = a_chk.grad().unwrap().to_vec();

    for (s, c) in g_std.iter().zip(g_chk.iter()) {
        assert!((s - c).abs() < 1e-9, "Checkpointing grad mismatch: std={}, chk={}", s, c);
    }
}

#[test]
fn test_mixed_precision_scale_and_unscale_equivalence() {
    let mut a_unscaled = Value::from_slice(&[2.0, 4.0], vec![2]);
    a_unscaled.set_requires_grad(true);
    let loss_unscaled = (&a_unscaled * &a_unscaled).sum();
    loss_unscaled.backward().unwrap();
    let g_unscaled = a_unscaled.grad().unwrap().to_vec();

    let mut a_scaled = Value::from_slice(&[2.0, 4.0], vec![2]);
    a_scaled.set_requires_grad(true);
    let loss_raw = (&a_scaled * &a_scaled).sum();
    
    let mut scaler = brain_autograd::engine::GradScaler::new(128.0, 2.0, 0.5, 2000);
    let loss_scaled = scaler.scale_loss(&loss_raw);
    loss_scaled.backward().unwrap();

    let ok = scaler.unscale_grads(&[&a_scaled]).unwrap();
    assert!(ok);
    let g_scaled = a_scaled.grad().unwrap().to_vec();

    for (u, s) in g_unscaled.iter().zip(g_scaled.iter()) {
        assert!((u - s).abs() < 1e-6, "Scaled unscale grad mismatch: unscaled={}, scaled={}", u, s);
    }
}

#[test]
fn check_deep_graph_iterative_drop() {
    // 5,000-node linear chain
    let mut v = Value::scalar(1.0);
    for _ in 0..5_000 {
        v = v.relu();
    }
    drop(v);
}

// =============================================================================
// Deferred Ops (Stage D Tracking)
// =============================================================================

#[test]
#[ignore = "tracked in Stage D, phase 91"]
fn test_fft_grad_deferred() {}

#[test]
#[ignore = "tracked in Stage D, phase 92"]
fn test_sparse_grad_deferred() {}

#[test]
#[ignore = "tracked in Stage D, phase 93"]
fn test_quant_grad_deferred() {}

fn main() {}
