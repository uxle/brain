//! # Loss Configurations
//!
//! Master loss configuration, hyperparameter specifications, and validation.
#![allow(missing_docs)]

use crate::core::{LossKind, Reduction};

/// General configuration for loss calculation.
#[derive(Debug, Clone)]
pub struct LossConfig {
    pub kind: LossKind,
    pub reduction: Reduction,
    pub label_smoothing: f64,
    pub gamma: f64,
    pub alpha: f64,
    pub delta: f64,
    pub temperature: f64,
    pub margin: f64,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            kind: LossKind::CrossEntropy,
            reduction: Reduction::Mean,
            label_smoothing: 0.0,
            gamma: 2.0,
            alpha: 0.25,
            delta: 1.0,
            temperature: 0.07,
            margin: 0.5,
        }
    }
}

impl LossConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.label_smoothing < 0.0 || self.label_smoothing >= 1.0 {
            return Err("label_smoothing must be in [0.0, 1.0)".into());
        }
        if self.gamma < 0.0 {
            return Err("gamma must be >= 0.0".into());
        }
        if self.delta <= 0.0 {
            return Err("delta must be > 0.0".into());
        }
        if self.temperature <= 0.0 {
            return Err("temperature must be > 0.0".into());
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "LossConfig[kind={:?} reduction={:?} smooth={:.2} gamma={:.2} delta={:.2} temp={:.2} margin={:.2}]",
            self.kind, self.reduction, self.label_smoothing, self.gamma, self.delta, self.temperature, self.margin
        )
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
