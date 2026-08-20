//! # Continuous Optimization Benchmarks
//!
//! Sphere, Rosenbrock, Rastrigin, Ackley, and Schwefel benchmark functions with known global minima.
#![allow(missing_docs)]

use std::f64::consts::PI;

/// Computes Sphere function: f(x) = sum(x_i^2). Global minimum f(0, ..., 0) = 0.
pub fn sphere_fn(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

/// Computes Rosenbrock function: f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2). Minimum f(1, ..., 1) = 0.
pub fn rosenbrock_fn(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for i in 0..x.len() - 1 {
        let term1 = x[i + 1] - x[i] * x[i];
        let term2 = 1.0 - x[i];
        sum += 100.0 * term1 * term1 + term2 * term2;
    }
    sum
}

/// Computes Rastrigin function: f(x) = 10 * d + sum(x_i^2 - 10 * cos(2 * pi * x_i)). Minimum f(0, ..., 0) = 0.
pub fn rastrigin_fn(x: &[f64]) -> f64 {
    let d = x.len() as f64;
    let sum: f64 = x.iter().map(|&v| v * v - 10.0 * (2.0 * PI * v).cos()).sum();
    10.0 * d + sum
}

/// Computes Ackley function: Minimum f(0, ..., 0) = 0.
pub fn ackley_fn(x: &[f64]) -> f64 {
    let d = x.len() as f64;
    if d == 0.0 {
        return 0.0;
    }

    let sum_sq: f64 = x.iter().map(|&v| v * v).sum();
    let sum_cos: f64 = x.iter().map(|&v| (2.0 * PI * v).cos()).sum();

    let term1 = -20.0 * (-0.2 * (sum_sq / d).sqrt()).exp();
    let term2 = -(sum_cos / d).exp();

    term1 + term2 + 20.0 + std::f64::consts::E
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
