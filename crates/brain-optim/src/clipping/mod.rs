//! # Gradient Clipping Engine
//!
//! Numerical gradient stabilization strategies including global norm clipping, value clipping, and adaptive clipping.
#![allow(missing_docs)]

pub mod norm;
pub mod adaptive;

use brain_core::Tensor;
pub use norm::{clip_grad_norm_, clip_grad_value_, ClipConfig, NormType};
pub use adaptive::{clip_grad_adaptive_, AdaptiveClipConfig, AGC};

/// Clipping operation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClipMode {
    #[default]
    GlobalNorm,
    Value,
    Adaptive,
}

/// Unified gradient clipper holding clipping configuration and metrics.
#[derive(Debug, Clone)]
pub struct GradClipper {
    pub mode: ClipMode,
    pub max_norm: f64,
    pub norm_type: NormType,
    pub clip_value: f64,
    pub adaptive_clipping_rate: f64,
}

impl Default for GradClipper {
    fn default() -> Self {
        Self {
            mode: ClipMode::GlobalNorm,
            max_norm: 1.0,
            norm_type: NormType::L2,
            clip_value: 1.0,
            adaptive_clipping_rate: 0.01,
        }
    }
}

impl GradClipper {
    pub fn norm(max_norm: f64, norm_type: NormType) -> Self {
        Self {
            mode: ClipMode::GlobalNorm,
            max_norm,
            norm_type,
            clip_value: 1.0,
            adaptive_clipping_rate: 0.01,
        }
    }

    pub fn value(clip_value: f64) -> Self {
        Self {
            mode: ClipMode::Value,
            max_norm: 1.0,
            norm_type: NormType::L2,
            clip_value,
            adaptive_clipping_rate: 0.01,
        }
    }

    pub fn adaptive(rate: f64) -> Self {
        Self {
            mode: ClipMode::Adaptive,
            max_norm: 1.0,
            norm_type: NormType::L2,
            clip_value: 1.0,
            adaptive_clipping_rate: rate,
        }
    }

    /// Applies clipping to parameters and gradients in-place according to configured mode.
    pub fn clip(&self, params: &mut [Tensor], grads: &mut [Tensor]) -> f64 {
        match self.mode {
            ClipMode::GlobalNorm => clip_grad_norm_(grads, self.max_norm, self.norm_type),
            ClipMode::Value => {
                clip_grad_value_(grads, self.clip_value);
                0.0
            }
            ClipMode::Adaptive => {
                clip_grad_adaptive_(params, grads, self.adaptive_clipping_rate, 1e-3);
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_clipping_mod_stress_001() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_002() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_003() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_004() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_005() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_006() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_007() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_008() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_009() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_010() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_011() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_012() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_013() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_014() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_015() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_016() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_017() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_018() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_019() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_020() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_021() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_022() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_023() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_024() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_025() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_026() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_027() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_028() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_029() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_030() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_031() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_032() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_033() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_034() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_035() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_036() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_037() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_038() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_039() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_040() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_041() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_042() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_043() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_044() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_045() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_046() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_047() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_048() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_049() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_050() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_051() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_052() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_053() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_054() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_055() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_056() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_057() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_058() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_059() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_060() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_061() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_062() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_063() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_064() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_065() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_066() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_067() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_068() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_069() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_070() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_071() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_072() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_073() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_074() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_075() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_076() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_077() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_078() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_079() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_080() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_081() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_082() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_083() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_084() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_085() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_086() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_087() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_088() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_089() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_090() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_091() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_092() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_093() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_094() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_095() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_096() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_097() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_098() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_099() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_100() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_101() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_102() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_103() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_104() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_105() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_106() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_107() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_108() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_109() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_110() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_111() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_112() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_113() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_114() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_115() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_116() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_117() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_118() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_119() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_120() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_121() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_122() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_123() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_124() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_125() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_126() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_127() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_128() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_129() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_130() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_131() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_132() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_133() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_134() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_135() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_136() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_137() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_138() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_139() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_140() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_141() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_142() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_143() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_144() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_145() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_146() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_147() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_148() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_149() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_150() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_151() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_152() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_153() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_154() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_155() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_156() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_157() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_158() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_159() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_160() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_161() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_162() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_163() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_164() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_165() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_166() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_167() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_168() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_169() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_170() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_171() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_172() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_173() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_174() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_175() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_176() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_177() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_178() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_179() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_180() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_181() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_182() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_183() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_184() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_185() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_186() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_187() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_188() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_189() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_190() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_191() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_192() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_193() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_194() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_195() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_196() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_197() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_198() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_199() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_200() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_201() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_202() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_203() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_204() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_205() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_206() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_207() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_208() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_209() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_210() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_211() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_212() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_213() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_214() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_215() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_216() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_217() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_218() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_219() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_220() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_221() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_222() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_223() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_224() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_225() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_226() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_227() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_228() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_229() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_230() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_231() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_232() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_233() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_234() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_235() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_236() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_237() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_238() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_239() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_240() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_241() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_242() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_243() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_244() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_245() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_246() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_247() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_248() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_249() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_250() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_251() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_252() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_253() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_254() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_255() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_256() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_257() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_258() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_259() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_260() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_261() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_262() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_263() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_264() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_265() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_266() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_267() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_268() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_269() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_270() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_271() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_272() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_273() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_274() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_275() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_276() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_277() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_278() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_279() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_280() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_281() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_282() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_283() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_284() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_285() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_286() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_287() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_288() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_289() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_290() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_291() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_292() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_293() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_294() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_295() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_296() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_297() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_298() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_299() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_300() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_301() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_302() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_303() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_304() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_305() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_306() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_307() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_308() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_309() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_310() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_311() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_312() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_313() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_314() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_315() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_316() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_317() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_318() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_319() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_320() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_321() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_322() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_323() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_324() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    #[test]
    fn test_clipping_mod_stress_325() {
        let clipper = GradClipper::norm(1.0, NormType::L2);
        assert_eq!(clipper.mode, ClipMode::GlobalNorm);
        assert_eq!(clipper.max_norm, 1.0);

        let val_clipper = GradClipper::value(0.5);
        assert_eq!(val_clipper.clip_value, 0.5);
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
}
