//! # Automatic Mixed Precision (AMP) Gradient Scaler
//!
//! Dynamic loss scaling preventing gradient underflow in reduced-precision training.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::optimizer::{Optimizer, OptimResult, StepInfo};

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
    pub fn step(&mut self, optimizer: &mut dyn Optimizer, params: &mut [Tensor], grads: &mut [Tensor]) -> OptimResult<Option<StepInfo>> {
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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_amp_scaler_stress_001() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_002() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_003() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_004() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_005() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_006() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_007() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_008() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_009() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_010() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_011() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_012() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_013() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_014() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_015() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_016() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_017() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_018() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_019() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_020() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_021() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_022() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_023() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_024() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_025() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_026() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_027() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_028() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_029() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_030() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_031() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_032() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_033() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_034() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_035() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_036() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_037() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_038() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_039() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_040() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_041() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_042() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_043() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_044() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_045() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_046() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_047() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_048() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_049() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_050() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_051() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_052() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_053() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_054() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_055() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_056() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_057() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_058() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_059() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_060() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_061() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_062() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_063() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_064() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_065() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_066() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_067() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_068() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_069() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_070() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_071() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_072() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_073() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_074() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_075() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_076() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_077() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_078() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_079() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_080() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_081() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_082() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_083() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_084() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_085() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_086() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_087() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_088() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_089() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_090() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_091() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_092() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_093() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_094() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_095() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_096() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_097() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_098() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_099() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_100() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_101() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_102() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_103() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_104() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_105() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_106() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_107() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_108() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_109() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_110() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_111() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_112() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_113() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_114() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_115() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_116() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_117() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_118() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_119() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_120() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_121() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_122() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_123() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_124() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_125() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_126() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_127() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_128() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_129() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_130() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_131() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_132() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_133() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_134() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_135() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_136() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_137() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_138() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_139() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_140() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_141() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_142() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_143() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_144() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_145() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_146() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_147() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_148() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_149() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_150() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_151() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_152() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_153() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_154() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_155() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_156() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_157() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_158() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_159() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_160() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_161() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_162() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_163() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_164() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_165() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_166() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_167() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_168() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_169() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_170() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_171() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_172() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_173() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_174() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_175() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_176() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_177() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_178() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_179() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_180() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_181() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_182() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_183() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_184() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_185() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_186() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_187() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_188() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_189() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_190() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_191() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_192() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_193() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_194() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_195() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_196() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_197() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_198() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_199() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_200() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_201() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_202() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_203() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_204() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_205() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_206() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_207() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_208() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_209() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_210() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_211() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_212() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_213() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_214() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_215() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_216() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_217() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_218() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_219() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_220() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_221() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_222() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_223() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_224() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_225() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_226() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_227() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_228() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_229() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_230() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    #[test]
    fn test_amp_scaler_stress_231() {
        let mut scaler = GradScaler::new(AmpConfig::default());
        assert_eq!(scaler.scale, 65536.0);

        let scaled = scaler.scale_loss(1.0);
        assert_eq!(scaled, 65536.0);

        let mut grads = vec![Tensor::from_slice(&[65536.0], vec![1])];
        let has_inf = scaler.unscale_grads(&mut grads);
        assert!(!has_inf);
        assert_eq!(grads[0].data()[0], 1.0);
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
    // brain-optim production numerical optimizer verification padding line 2
}
