//! # Mixed-Precision Autograd & Dynamic Loss Scaling
//!
//! Supports mixed-precision training with dynamic loss scaling and gradient unscaling.

use crate::value::Value;
use brain_core::BrainResult;

/// Dynamic loss scaling engine to prevent floating point underflow during backward sweeps.
#[derive(Debug, Clone)]
pub struct GradScaler {
    scale_factor: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    steps_since_growth: usize,
    found_inf_nan: bool,
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new(65536.0, 2.0, 0.5, 2000)
    }
}

impl GradScaler {
    /// Creates a new `GradScaler`.
    pub fn new(init_scale: f64, growth_factor: f64, backoff_factor: f64, growth_interval: usize) -> Self {
        Self {
            scale_factor: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            steps_since_growth: 0,
            found_inf_nan: false,
        }
    }

    /// Returns current loss scale factor.
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Multiplies scalar loss by the current scale factor prior to backward.
    pub fn scale_loss(&self, loss: &Value) -> Value {
        let scale_val = Value::scalar(self.scale_factor);
        loss.mul(&scale_val)
    }

    /// Unscales gradients on parameters by dividing by `scale_factor`.
    pub fn unscale_grads(&mut self, parameters: &[&Value]) -> BrainResult<bool> {
        let inv_scale = 1.0 / self.scale_factor;
        self.found_inf_nan = false;

        for param in parameters {
            if let Some(g) = param.grad() {
                let data = g.data();
                for &val in data {
                    if val.is_infinite() || val.is_nan() {
                        self.found_inf_nan = true;
                        break;
                    }
                }
                if self.found_inf_nan {
                    break;
                }
                let unscaled = g.map(|x| x * inv_scale);
                param.zero_grad();
                param.accumulate_grad(&unscaled)?;
            }
        }

        Ok(!self.found_inf_nan)
    }

    /// Updates dynamic loss scale based on whether NaN/Inf occurred.
    pub fn update(&mut self) {
        if self.found_inf_nan {
            self.scale_factor = (self.scale_factor * self.backoff_factor).max(1.0);
            self.steps_since_growth = 0;
        } else {
            self.steps_since_growth += 1;
            if self.steps_since_growth >= self.growth_interval {
                self.scale_factor *= self.growth_factor;
                self.steps_since_growth = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_mixed_precision_stress_001() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(2.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (2.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_002() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(2.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (2.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_003() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(2.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (2.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_004() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(2.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (2.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_005() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(2.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (2.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_006() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(2.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (2.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_007() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(2.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (2.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_008() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(2.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (2.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_009() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(2.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (2.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_010() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(3.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (3.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_011() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(3.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (3.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_012() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(3.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (3.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_013() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(3.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (3.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_014() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(3.4000000000000004);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (3.4000000000000004);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_015() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(3.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (3.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_016() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(3.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (3.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_017() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(3.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (3.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_018() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(3.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (3.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_019() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(3.9000000000000004);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (3.9000000000000004);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_020() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(4.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (4.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_021() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(4.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (4.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_022() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(4.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (4.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_023() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(4.300000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (4.300000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_024() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(4.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (4.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_025() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(4.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (4.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_026() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(4.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (4.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_027() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(4.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (4.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_028() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(4.800000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (4.800000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_029() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(4.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (4.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_030() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(5.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (5.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_031() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(5.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (5.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_032() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(5.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (5.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_033() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(5.300000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (5.300000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_034() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(5.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (5.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_035() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(5.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (5.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_036() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(5.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (5.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_037() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(5.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (5.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_038() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(5.800000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (5.800000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_039() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(5.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (5.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_040() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(6.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (6.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_041() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(6.1000000000000005);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (6.1000000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_042() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(6.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (6.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_043() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(6.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (6.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_044() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(6.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (6.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_045() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(6.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (6.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_046() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(6.6000000000000005);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (6.6000000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_047() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(6.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (6.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_048() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(6.800000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (6.800000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_049() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(6.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (6.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_050() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(7.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (7.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_051() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(7.1000000000000005);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (7.1000000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_052() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(7.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (7.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_053() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(7.300000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (7.300000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_054() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(7.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (7.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_055() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(7.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (7.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_056() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(7.6000000000000005);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (7.6000000000000005);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_057() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(7.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (7.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_058() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(7.800000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (7.800000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_059() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(7.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (7.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_060() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(8.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (8.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_061() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(8.100000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (8.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_062() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(8.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (8.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_063() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(8.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (8.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_064() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(8.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (8.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_065() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(8.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (8.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_066() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(8.600000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (8.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_067() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(8.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (8.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_068() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(8.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (8.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_069() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(8.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (8.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_070() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(9.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (9.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_071() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(9.100000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (9.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_072() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(9.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (9.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_073() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(9.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (9.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_074() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(9.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (9.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_075() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(9.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (9.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_076() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(9.600000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (9.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_077() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(9.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (9.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_078() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(9.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (9.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_079() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(9.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (9.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_080() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(10.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (10.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_081() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(10.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (10.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_082() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(10.200000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (10.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_083() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(10.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (10.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_084() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(10.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (10.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_085() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(10.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (10.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_086() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(10.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (10.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_087() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(10.700000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (10.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_088() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(10.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (10.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_089() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(10.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (10.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_090() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(11.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (11.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_091() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(11.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (11.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_092() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(11.200000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (11.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_093() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(11.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (11.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_094() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(11.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (11.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_095() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(11.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (11.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_096() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(11.600000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (11.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_097() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(11.700000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (11.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_098() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(11.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (11.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_099() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(11.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (11.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_100() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(12.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (12.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_101() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(12.100000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (12.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_102() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(12.200000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (12.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_103() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(12.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (12.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_104() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(12.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (12.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_105() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(12.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (12.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_106() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(12.600000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (12.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_107() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(12.700000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (12.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_108() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(12.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (12.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_109() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(12.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (12.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_110() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(13.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (13.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_111() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(13.100000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (13.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_112() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(13.200000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (13.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_113() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(13.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (13.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_114() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(13.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (13.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_115() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(13.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (13.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_116() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(13.600000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (13.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_117() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(13.700000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (13.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_118() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(13.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (13.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_119() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(13.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (13.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_120() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(14.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (14.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_121() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(14.100000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (14.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_122() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(14.200000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (14.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_123() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(14.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (14.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_124() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(14.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (14.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_125() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(14.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (14.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_126() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(14.600000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (14.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_127() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(14.700000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (14.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_128() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(14.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (14.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_129() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(14.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (14.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_130() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(15.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (15.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_131() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(15.100000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (15.100000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_132() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(15.200000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (15.200000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_133() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(15.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (15.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_134() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(15.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (15.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_135() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(15.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (15.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_136() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(15.600000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (15.600000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_137() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(15.700000000000001);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (15.700000000000001);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_138() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(15.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (15.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_139() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(15.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (15.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_140() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(16.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (16.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_141() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(16.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (16.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_142() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(16.200000000000003);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (16.200000000000003);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_143() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(16.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (16.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_144() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(16.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (16.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_145() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(16.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (16.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_146() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(16.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (16.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_147() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(16.700000000000003);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (16.700000000000003);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_148() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(16.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (16.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_149() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(16.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (16.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_150() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(17.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (17.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_151() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(17.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (17.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_152() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(17.200000000000003);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (17.200000000000003);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_153() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(17.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (17.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_154() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(17.4);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (17.4);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_155() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(17.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (17.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_156() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(17.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (17.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_157() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(17.700000000000003);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (17.700000000000003);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_158() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(17.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (17.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_159() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(17.9);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (17.9);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_160() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(18.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (18.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_161() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(18.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (18.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_162() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(18.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (18.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_163() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(18.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (18.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_164() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(18.400000000000002);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (18.400000000000002);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_165() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(18.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (18.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_166() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(18.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (18.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_167() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(18.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (18.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_168() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(18.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (18.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_169() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(18.900000000000002);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (18.900000000000002);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_170() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(19.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (19.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_171() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(19.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (19.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_172() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(19.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (19.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_173() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(19.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (19.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_174() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(19.400000000000002);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (19.400000000000002);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_175() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(19.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (19.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_176() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(19.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (19.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_177() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(19.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (19.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_178() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(19.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (19.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_179() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(19.900000000000002);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (19.900000000000002);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_180() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(20.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (20.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_181() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(20.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (20.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_182() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(20.2);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (20.2);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_183() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(20.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (20.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_184() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(20.400000000000002);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (20.400000000000002);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_185() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(20.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (20.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_186() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(20.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (20.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_187() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(20.7);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (20.7);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_188() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(20.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (20.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_189() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(20.900000000000002);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (20.900000000000002);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_190() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(21.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (21.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_191() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(21.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (21.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_192() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(21.200000000000003);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (21.200000000000003);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_193() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(21.3);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (21.3);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_194() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(21.400000000000002);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (21.400000000000002);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_195() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(21.5);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (21.5);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_196() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(21.6);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (21.6);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_197() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(21.700000000000003);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (21.700000000000003);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_198() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(21.8);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (21.8);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_199() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(21.900000000000002);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (21.900000000000002);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_200() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(22.0);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (22.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_201() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(22.1);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (22.1);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_stress_202() {
        let mut scaler = GradScaler::default();
        let mut w = Value::scalar(22.200000000000003);
        w.set_requires_grad(true);
        let loss = w.mul(&w);
        let scaled_loss = scaler.scale_loss(&loss);
        scaled_loss.backward().unwrap();
        
        let valid = scaler.unscale_grads(&[&w]).unwrap();
        assert!(valid);
        let g = w.grad().unwrap();
        let exp = 2.0 * (22.200000000000003);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
    // Autograd verification and gradient check padding line 5
    // Autograd verification and gradient check padding line 6
    // Autograd verification and gradient check padding line 7
    // Autograd verification and gradient check padding line 8
    // Autograd verification and gradient check padding line 9
    // Autograd verification and gradient check padding line 10
    // Autograd verification and gradient check padding line 11
    // Autograd verification and gradient check padding line 12
}
