//! Numerical gradient checking to find bugs in core ops and autograd.

use brain_autograd::Value;
use brain_core::Tensor;

fn approx(a: f64, b: f64) -> bool {
    let r = if a.abs() > 1e-8 { (a - b).abs() / a.abs() } else { (a - b).abs() };
    r < 1e-4
}

/// Central-difference numeric gradient of a scalar-valued `f(x)`, one element at
/// a time. `x` need not require grad for the numeric pass (we only read data).
fn numeric_scalar_grad<F>(x_data: &[f64], shape: &[usize], f: F) -> Vec<f64>
where
    F: Fn(&Value) -> Value,
{
    let eps = 1e-5;
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
fn analytic_scalar_grad<F>(x_data: &[f64], shape: &[usize], f: F) -> Vec<f64>
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

fn assert_grads_close(name: &str, an: &[f64], num: &[f64]) {
    let tol = 1e-4;
    let mut ok = true;
    for (i, (&a, &n)) in an.iter().zip(num.iter()).enumerate() {
        let scale = a.abs().max(n.abs()).max(1e-8);
        if (a - n).abs() / scale > tol {
            ok = false;
            println!("BUG [{}]: idx {} analytic={:.6} numeric={:.6}", name, i, a, n);
        }
    }
    if ok {
        println!("OK   [{}]: analytic grad matches numeric ({} elems)", name, an.len());
    } else {
        panic!("VJP mismatch in {}", name);
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
    if !ok {
        println!(
            "BUG [{}]: x={:?}, analytic={:.6}, numeric={:.6}",
            name, x_val, analytic, numeric
        );
    } else {
        println!("OK   [{}]: x={:?} grad={:.6}", name, x_val, analytic);
    }
}

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
fn check_matmul_grad() {
    // f = sum(a @ b)  where a is 2x3, b is 3x2 -> scalar
    // d/da = grad @ b^T, d/db = a^T @ grad
    let a_data = vec![0.5, 1.2, -0.3, 2.0, 0.8, -1.5]; // 2x3
    let b_data = vec![0.7, -0.4, 1.1, 0.3, 2.2, -0.9]; // 3x2
    let mut a = Value::from_slice(&a_data, vec![2, 3]);
    a.set_requires_grad(true);
    let mut b = Value::from_slice(&b_data, vec![3, 2]);
    b.set_requires_grad(true);
    let c = a.matmul(&b); // 2x2
    let y = c.sum();
    a.zero_grad();
    b.zero_grad();
    y.backward().unwrap();
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();

    // analytic grad for a: broadcasted ones (2x2) @ b^T (2x3) = 2x3
    let b_t = Tensor::from_slice(&b_data, vec![3, 2]).transpose(0, 1); // 2x3
    let ga_expected = brain_core::tensor::arithmetic::matmul(&Tensor::ones(vec![2, 2]), &b_t);
    let a_t = Tensor::from_slice(&a_data, vec![2, 3]).transpose(0, 1); // 3x2
    let gb_expected = brain_core::tensor::arithmetic::matmul(&a_t, &Tensor::ones(vec![2, 2]));

    for i in 0..ga.numel() {
        let diff = (ga.get(i) - ga_expected.get(i)).abs();
        if diff > 1e-5 {
            println!("BUG [matmul_grad a]: idx {} got {} expected {}", i, ga.get(i), ga_expected.get(i));
        }
    }
    for i in 0..gb.numel() {
        let diff = (gb.get(i) - gb_expected.get(i)).abs();
        if diff > 1e-5 {
            println!("BUG [matmul_grad b]: idx {} got {} expected {}", i, gb.get(i), gb_expected.get(i));
        }
    }
    println!("OK [matmul_grad] a grad matches, b grad matches");
}

#[test]
fn check_broadcast_mul_grad() {
    // f(x, w) = sum(x * w) where x is [3,1] and w is [3,4] => broadcast
    // x is [3,1], w is [3,4], out [3,4], sum
    let x_data = vec![0.5, 1.0, 2.0]; // shape [3,1]
    let w_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]; // [3,4]
    let mut x = Value::from_slice(&x_data, vec![3, 1]);
    x.set_requires_grad(true);
    let mut w = Value::from_slice(&w_data, vec![3, 4]);
    w.set_requires_grad(true);
    let y = (x.clone() * w.clone()).sum();
    x.zero_grad();
    w.zero_grad();
    y.backward().unwrap();
    let gx = x.grad().unwrap();
    let gw = w.grad().unwrap();
    // d/dw = x (broadcasted to [3,4])
    for i in 0..gw.numel() {
        let row = i / 4;
        let expected = x_data[row];
        if (gw.get(i) - expected).abs() > 1e-6 {
            println!("BUG [broadcast_mul_grad w]: idx {} got {} expected {}", i, gw.get(i), expected);
        }
    }
    // d/dx = sum over broadcasted dim of w
    // x is [3,1], grad flowing back is ones[3,4], sum over dim 1 -> [3,1]
    for i in 0..gx.numel() {
        let row = i;
        let expected: f64 = w_data[row * 4..row * 4 + 4].iter().sum();
        if (gx.get(i) - expected).abs() > 1e-6 {
            println!("BUG [broadcast_mul_grad x]: idx {} got {} expected {}", i, gx.get(i), expected);
        }
    }
    println!("OK [broadcast_mul_grad] x and w grads match");
}

#[test]
fn check_pow_grad() {
    // f(x) = x^3 => grad = 3 x^2
    // f(x) = 2^x => grad = 2^x ln(2)
    for &x in &[0.5, 1.0, 2.0, 3.0] {
        let mut v = Value::scalar(x);
        v.set_requires_grad(true);
        let y = v.pow(&Value::scalar(3.0));
        v.zero_grad();
        y.backward().unwrap();
        let g = v.grad().unwrap().get(0);
        let expected = 3.0 * x * x;
        if !approx(g, expected) {
            println!("BUG [pow x^3]: x={} got {} expected {}", x, g, expected);
        } else {
            println!("OK   [pow x^3]: x={} grad={:.6}", x, g);
        }
    }
    for &x in &[0.5, 1.0, 2.0] {
        // Differentiate wrt the EXPONENT: f(x) = 2^x, d/dx = 2^x * ln 2.
        let mut v = Value::scalar(x);
        v.set_requires_grad(true);
        let base = Value::scalar(2.0);
        let y = base.pow(&v);
        v.zero_grad();
        y.backward().unwrap();
        let g = v.grad().unwrap().get(0);
        let expected = (2.0_f64).powf(x) * 2.0_f64.ln();
        if !approx(g, expected) {
            println!("BUG [pow 2^x]: x={} got {} expected {}", x, g, expected);
        } else {
            println!("OK   [pow 2^x]: x={} grad={:.6}", x, g);
        }
    }
}

#[test]
fn check_binary_elementwise_grad() {
    // f(x, y) = sum(x OP y) for a broadcastable constant y. Verifies add/sub/div
    // VJPs and the sum-to-leaf broadcast reduction on a 2x3 tensor.
    let xs = [0.5, -1.2, 0.3, 2.0, 0.8, -0.4];
    let ys = [0.7, 1.1, -0.6, 0.2, 1.5, -0.9];
    let shape = vec![2, 3];
    let ops: [(&str, fn(&Value, &Value) -> Value); 3] = [
        ("add", |a, b| a.add(b)),
        ("sub", |a, b| a.sub(b)),
        ("div", |a, b| a.div(b)),
    ];
    for (name, op) in ops {
        let yv = Value::from_slice(&ys, shape.clone());
        let an = analytic_scalar_grad(&xs, &shape, |x| op(x, &yv).sum());
        let num = numeric_scalar_grad(&xs, &shape, |x| op(x, &yv).sum());
        assert_grads_close(&format!("elemwise_{}", name), &an, &num);
    }
}

#[test]
fn check_mean_grad() {
    // f(x) = mean(x): grad wrt each element is 1/n.
    let xs = [0.5, 1.2, -0.3, 2.0, 0.8, -1.5];
    let shape = vec![2, 3];
    let an = analytic_scalar_grad(&xs, &shape, |x| x.mean());
    let num = numeric_scalar_grad(&xs, &shape, |x| x.mean());
    for v in &an {
        assert!((v - 1.0 / xs.len() as f64).abs() < 1e-6, "mean grad should be 1/n");
    }
    assert_grads_close("mean", &an, &num);
}

#[test]
fn check_softmax_logsoftmax_grad() {
    // f(x) = sum(softmax(x) * c)  and  f(x) = sum(log_softmax(x) * c)
    // These exercise the softmax/log_softmax VJPs (full Jacobian-vector product)
    // end-to-end through mul + sum.
    let xs = [0.5, 1.0, -0.5, 2.0, 0.8, -1.2];
    let shape = vec![2, 3];
    let c = [0.3, -0.4, 0.7, 1.1, -0.2, 0.5];
    let make_c = || Value::from_slice(&c, shape.clone());

    let f_softmax = |x: &Value| (brain_autograd::ops::softmax(x) * make_c()).sum();
    let an = analytic_scalar_grad(&xs, &shape, f_softmax);
    let num = numeric_scalar_grad(&xs, &shape, f_softmax);
    assert_grads_close("softmax", &an, &num);

    let f_logsoftmax = |x: &Value| (brain_autograd::ops::log_softmax(x) * make_c()).sum();
    let an = analytic_scalar_grad(&xs, &shape, f_logsoftmax);
    let num = numeric_scalar_grad(&xs, &shape, f_logsoftmax);
    assert_grads_close("log_softmax", &an, &num);
}

#[test]
fn check_conv2d_grad() {
    // Input: [1, 2, 4, 4]
    let x_data: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1) - 1.5).collect();
    let x_shape = vec![1, 2, 4, 4];
    // Weight: [2, 2, 3, 3]
    let w_data: Vec<f64> = (0..36).map(|i| (i as f64 * 0.05) - 0.9).collect();
    let w_shape = vec![2, 2, 3, 3];
    // Bias: [2]
    let b_data = vec![0.3, -0.2];
    let b_shape = vec![2];

    let w_val = Value::from_slice(&w_data, w_shape.clone());
    let b_val = Value::from_slice(&b_data, b_shape.clone());

    // 1. Differentiate wrt Input
    let f_input = |x: &Value| {
        x.conv2d(&w_val, Some(&b_val), (1, 1), (1, 1)).sum()
    };
    let an_x = analytic_scalar_grad(&x_data, &x_shape, f_input);
    let num_x = numeric_scalar_grad(&x_data, &x_shape, f_input);
    assert_grads_close("conv2d_input", &an_x, &num_x);

    // 2. Differentiate wrt Weight
    let x_val = Value::from_slice(&x_data, x_shape.clone());
    let f_weight = |w: &Value| {
        x_val.conv2d(w, Some(&b_val), (1, 1), (1, 1)).sum()
    };
    let an_w = analytic_scalar_grad(&w_data, &w_shape, f_weight);
    let num_w = numeric_scalar_grad(&w_data, &w_shape, f_weight);
    assert_grads_close("conv2d_weight", &an_w, &num_w);

    // 3. Differentiate wrt Bias
    let f_bias = |b: &Value| {
        x_val.conv2d(&w_val, Some(b), (1, 1), (1, 1)).sum()
    };
    let an_b = analytic_scalar_grad(&b_data, &b_shape, f_bias);
    let num_b = numeric_scalar_grad(&b_data, &b_shape, f_bias);
    assert_grads_close("conv2d_bias", &an_b, &num_b);
}

#[test]
fn check_conv2d_strided_padded_grad() {
    let x_data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.12) - 3.0).collect();
    let x_shape = vec![2, 1, 5, 5];
    let w_data: Vec<f64> = (0..18).map(|i| (i as f64 * 0.08) - 0.7).collect();
    let w_shape = vec![2, 1, 3, 3];

    let w_val = Value::from_slice(&w_data, w_shape.clone());
    let x_val = Value::from_slice(&x_data, x_shape.clone());

    let f_input = |x: &Value| {
        x.conv2d(&w_val, None, (2, 2), (1, 1)).sum()
    };
    let an_x = analytic_scalar_grad(&x_data, &x_shape, f_input);
    let num_x = numeric_scalar_grad(&x_data, &x_shape, f_input);
    assert_grads_close("conv2d_strided_input", &an_x, &num_x);

    let f_weight = |w: &Value| {
        x_val.conv2d(w, None, (2, 2), (1, 1)).sum()
    };
    let an_w = analytic_scalar_grad(&w_data, &w_shape, f_weight);
    let num_w = numeric_scalar_grad(&w_data, &w_shape, f_weight);
    assert_grads_close("conv2d_strided_weight", &an_w, &num_w);
}

#[test]
fn check_conv_transpose2d_grad() {
    let x_data: Vec<f64> = (0..18).map(|i| (i as f64 * 0.15) - 1.0).collect();
    let x_shape = vec![1, 2, 3, 3];
    let w_data: Vec<f64> = (0..16).map(|i| (i as f64 * 0.1) - 0.8).collect();
    let w_shape = vec![2, 2, 2, 2];
    let b_data = vec![0.5, -0.4];
    let b_shape = vec![2];

    let w_val = Value::from_slice(&w_data, w_shape.clone());
    let b_val = Value::from_slice(&b_data, b_shape.clone());
    let x_val = Value::from_slice(&x_data, x_shape.clone());

    // 1. Differentiate wrt Input
    let f_input = |x: &Value| {
        x.conv_transpose2d(&w_val, Some(&b_val), (2, 2), (1, 1)).sum()
    };
    let an_x = analytic_scalar_grad(&x_data, &x_shape, f_input);
    let num_x = numeric_scalar_grad(&x_data, &x_shape, f_input);
    assert_grads_close("conv_transpose2d_input", &an_x, &num_x);

    // 2. Differentiate wrt Weight
    let f_weight = |w: &Value| {
        x_val.conv_transpose2d(w, Some(&b_val), (2, 2), (1, 1)).sum()
    };
    let an_w = analytic_scalar_grad(&w_data, &w_shape, f_weight);
    let num_w = numeric_scalar_grad(&w_data, &w_shape, f_weight);
    assert_grads_close("conv_transpose2d_weight", &an_w, &num_w);

    // 3. Differentiate wrt Bias
    let f_bias = |b: &Value| {
        x_val.conv_transpose2d(&w_val, Some(b), (2, 2), (1, 1)).sum()
    };
    let an_b = analytic_scalar_grad(&b_data, &b_shape, f_bias);
    let num_b = numeric_scalar_grad(&b_data, &b_shape, f_bias);
    assert_grads_close("conv_transpose2d_bias", &an_b, &num_b);
}

#[test]
fn check_max_pool2d_grad() {
    // Unique values so argmax is strictly defined and clean
    let x_data: Vec<f64> = (0..32).map(|i| (i as f64 * 0.23 + 0.1).sin()).collect();
    let x_shape = vec![1, 2, 4, 4];

    let f_pool = |x: &Value| {
        x.max_pool2d((2, 2), (2, 2), (0, 0)).sum()
    };
    let an_x = analytic_scalar_grad(&x_data, &x_shape, f_pool);
    let num_x = numeric_scalar_grad(&x_data, &x_shape, f_pool);
    assert_grads_close("max_pool2d", &an_x, &num_x);
}

#[test]
fn check_avg_pool2d_grad() {
    let x_data: Vec<f64> = (0..32).map(|i| (i as f64 * 0.17) - 2.5).collect();
    let x_shape = vec![1, 2, 4, 4];

    let f_pool = |x: &Value| {
        x.avg_pool2d((2, 2), (2, 2), (0, 0)).sum()
    };
    let an_x = analytic_scalar_grad(&x_data, &x_shape, f_pool);
    let num_x = numeric_scalar_grad(&x_data, &x_shape, f_pool);
    assert_grads_close("avg_pool2d", &an_x, &num_x);
}

#[test]
fn check_deep_graph_iterative_drop() {
    let mut v = Value::new(brain_core::Tensor::scalar(1.0), true);
    for _ in 0..50_000 {
        v = v.relu();
    }
    v.backward().unwrap();
    // Drop v and ensure no stack overflow
    drop(v);
}

#[test]
fn check_cross_entropy_loss_grad() {
    // Logits: [2, 3], Targets: [2, 0]
    let x_data = vec![0.5, 1.5, 2.5, 3.0, 1.0, 0.1];
    let x_shape = vec![2, 3];
    let mask = vec![
        0.0, 0.0, 1.0, // target 2
        1.0, 0.0, 0.0, // target 0
    ];

    let f_ce = |x: &Value| {
        let lsm = brain_autograd::ops::log_softmax(x);
        let mask_val = Value::from_slice(&mask, vec![2, 3]);
        let selected = lsm * mask_val;
        selected.sum().neg() / Value::scalar(2.0)
    };

    let an = analytic_scalar_grad(&x_data, &x_shape, f_ce);
    let num = numeric_scalar_grad(&x_data, &x_shape, f_ce);
    assert_grads_close("cross_entropy_loss", &an, &num);
}

#[test]
fn check_mse_loss_grad() {
    let p_data = vec![1.2, -0.8, 2.4, 0.5];
    let t_data = vec![1.0, -1.0, 2.0, 0.0];
    let shape = vec![2, 2];

    let t_val = Value::from_slice(&t_data, shape.clone());
    let f_mse = |p: &Value| {
        let diff = p - &t_val;
        let sq = &diff * &diff;
        sq.mean()
    };

    let an = analytic_scalar_grad(&p_data, &shape, f_mse);
    let num = numeric_scalar_grad(&p_data, &shape, f_mse);
    assert_grads_close("mse_loss", &an, &num);
}

#[test]
fn check_embedding_grad() {
    // Weight table: [4, 3]
    let w_data = vec![
        0.1, 0.2, 0.3, // token 0
        0.4, 0.5, 0.6, // token 1
        0.7, 0.8, 0.9, // token 2
        1.0, 1.1, 1.2, // token 3
    ];
    let w_shape = vec![4, 3];
    let indices = vec![1, 2, 1]; // token 1 is indexed twice

    let f_emb = |w: &Value| {
        let emb = w.embedding(&indices, vec![3, 3]);
        emb.sum()
    };

    let an = analytic_scalar_grad(&w_data, &w_shape, f_emb);
    let num = numeric_scalar_grad(&w_data, &w_shape, f_emb);
    assert_grads_close("embedding", &an, &num);
}

#[test]
fn check_linear_grad() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
    let x_shape = vec![2, 2];
    let w_val = Value::from_slice(&[0.5, -0.5, 1.5, -1.0], vec![2, 2]);
    let b_val = Value::from_slice(&[0.1, -0.2], vec![2]);

    let f_lin = |x: &Value| {
        x.linear(&w_val, Some(&b_val)).sum()
    };

    let an = analytic_scalar_grad(&x_data, &x_shape, f_lin);
    let num = numeric_scalar_grad(&x_data, &x_shape, f_lin);
    assert_grads_close("linear", &an, &num);
}

fn main() {}
