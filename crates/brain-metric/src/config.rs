//! # Metric Configurations
//!
//! Master metric configuration, averaging modes, threshold settings, and validation.
#![allow(missing_docs)]

use crate::core::MetricKind;

/// Multiclass averaging reduction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AverageMode {
    #[default]
    Macro,
    Micro,
    Weighted,
    None,
}

/// General configuration for metric evaluations.
#[derive(Debug, Clone)]
pub struct MetricConfig {
    pub kind: MetricKind,
    pub average: AverageMode,
    pub top_k: usize,
    pub threshold: f64,
    pub num_classes: usize,
    pub iou_threshold: f64,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            kind: MetricKind::Accuracy,
            average: AverageMode::Macro,
            top_k: 1,
            threshold: 0.5,
            num_classes: 2,
            iou_threshold: 0.5,
        }
    }
}

impl MetricConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.top_k == 0 {
            return Err("top_k must be at least 1".into());
        }
        if self.threshold < 0.0 || self.threshold > 1.0 {
            return Err("threshold must be in [0.0, 1.0]".into());
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "MetricConfig[kind={:?} avg={:?} top_k={} thresh={:.2} iou={:.2}]",
            self.kind, self.average, self.top_k, self.threshold, self.iou_threshold
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
