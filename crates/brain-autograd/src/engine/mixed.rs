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
                        return Ok(false);
                    }
                }
            }
        }

        for param in parameters {
            if let Some(g) = param.grad() {
                let unscaled = g.map(|x| x * inv_scale);
                param.zero_grad();
                param.accumulate_grad(&unscaled)?;
            }
        }

        Ok(true)
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
}
