//! # Automatic Mixed Precision (AMP) Gradient Scaler
//!
//! Dynamic loss scaling preventing gradient underflow in reduced-precision training.
#![allow(missing_docs)]

use crate::optimizer::{OptimResult, Optimizer, StepInfo};
use brain_core::Tensor;

/// Configuration settings for dynamic loss scaling.
#[derive(Debug, Clone, PartialEq)]
pub struct AmpConfig {
    pub init_scale: f64,
    pub growth_factor: f64,
    pub backoff_factor: f64,
    pub growth_interval: usize,
    pub enabled: bool,
}

impl Default for AmpConfig {
    fn default() -> Self {
        Self {
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            enabled: true,
        }
    }
}

/// Dynamic Loss Scaler managing automatic mixed precision steps.
#[derive(Debug, Clone)]
pub struct GradScaler {
    pub config: AmpConfig,
    pub scale: f64,
    pub growth_tracker: usize,
    pub found_inf: bool,
}

impl GradScaler {
    pub fn new(config: AmpConfig) -> Self {
        let scale = config.init_scale;
        Self {
            config,
            scale,
            growth_tracker: 0,
            found_inf: false,
        }
    }

    /// Scales the loss value prior to backward pass.
    pub fn scale_loss(&self, loss: f64) -> f64 {
        if self.config.enabled {
            loss * self.scale
        } else {
            loss
        }
    }

    /// Unscales gradients in-place and checks for NaN / Infinity overflow.
    pub fn unscale_grads(&mut self, grads: &mut [Tensor]) -> bool {
        if !self.config.enabled {
            return false;
        }

        let inv_scale = 1.0 / self.scale;
        let mut has_overflow = false;

        for g in grads.iter_mut() {
            for val in g.data_mut() {
                if val.is_nan() || val.is_infinite() {
                    has_overflow = true;
                }
                *val *= inv_scale;
            }
        }

        self.found_inf = has_overflow;
        has_overflow
    }

    /// Advances optimizer step if no overflow occurred; adjusts loss scale factor accordingly.
    pub fn step(
        &mut self,
        optimizer: &mut dyn Optimizer,
        params: &mut [Tensor],
        grads: &mut [Tensor],
    ) -> OptimResult<Option<StepInfo>> {
        if !self.config.enabled {
            return optimizer.step(params, grads).map(Some);
        }

        let has_inf = self.unscale_grads(grads);

        if has_inf {
            self.scale *= self.config.backoff_factor;
            self.growth_tracker = 0;
            Ok(None)
        } else {
            let res = optimizer.step(params, grads)?;
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                self.scale *= self.config.growth_factor;
                self.growth_tracker = 0;
            }
            Ok(Some(res))
        }
    }
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
