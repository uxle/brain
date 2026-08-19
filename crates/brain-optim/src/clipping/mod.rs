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
}
