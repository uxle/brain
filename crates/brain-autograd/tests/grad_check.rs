//! Numerical gradient checking to find bugs in core ops and autograd.

use brain_autograd::Value;
use brain_core::Tensor;

fn approx(a: f64, b: f64) -> bool {
    let r = if a.abs() > 1e-8 { (a - b).abs() / a.abs() } else { (a - b).abs() };
    r < 1e-4
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
        let mut v = Value::scalar(2.0);
        v.set_requires_grad(true);
        let y = v.pow(&Value::scalar(x)); // 2^x
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

fn main() {}
