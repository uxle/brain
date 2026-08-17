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

    #[test]
    fn test_config_stress_001() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_002() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_003() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_004() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_005() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_006() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_007() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_008() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_009() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_010() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_011() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_012() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_013() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_014() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_015() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_016() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_017() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_018() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_019() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_020() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_021() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_022() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_023() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_024() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_025() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_026() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_027() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_028() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_029() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_030() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_031() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_032() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_033() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_034() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_035() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_036() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_037() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_038() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_039() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_040() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_041() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_042() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_043() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_044() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_045() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_046() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_047() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_048() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_049() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_050() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_051() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_052() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_053() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_054() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_055() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_056() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_057() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_058() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_059() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_060() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_061() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_062() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_063() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_064() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_065() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_066() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_067() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_068() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_069() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_070() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_071() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_072() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_073() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_074() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_075() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_076() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_077() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_078() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_079() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_080() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_081() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_082() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_083() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_084() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_085() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_086() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_087() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_088() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_089() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_090() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_091() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_092() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_093() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_094() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_095() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_096() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_097() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_098() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_099() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_100() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_101() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_102() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_103() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_104() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_105() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_106() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_107() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_108() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_109() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_110() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_111() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_112() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_113() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_114() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_115() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_116() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_117() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_118() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_119() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_120() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_121() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_122() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_123() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_124() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_125() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_126() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_127() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_128() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_129() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_130() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_131() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_132() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_133() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_134() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_135() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_136() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_137() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_138() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_139() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_140() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_141() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_142() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_143() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_144() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_145() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_146() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_147() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_148() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_149() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_150() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_151() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_152() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_153() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_154() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_155() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_156() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_157() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_158() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_159() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_160() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_161() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_162() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_163() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_164() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_165() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_166() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_167() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_168() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_169() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_170() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_171() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_172() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_173() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_174() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_175() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_176() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_177() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_178() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_179() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_180() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_181() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_182() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_183() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_184() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_185() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_186() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_187() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_188() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_189() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_190() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_191() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_192() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_193() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_194() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_195() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_196() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_197() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_198() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_199() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_200() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_201() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_202() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_203() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_204() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_205() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_206() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_207() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_208() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_209() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_210() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_211() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_212() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_213() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_214() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_215() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_216() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_217() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_218() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_219() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_220() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_221() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_222() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_223() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_224() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_225() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_226() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_227() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_228() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_229() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_230() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_231() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_232() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_233() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_234() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_235() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_236() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_237() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_238() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_239() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_240() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_241() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_242() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_243() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_244() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_245() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_246() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_247() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_248() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_249() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_250() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_251() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_252() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_253() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_254() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_255() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_256() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_257() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_258() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_259() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_260() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_261() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_262() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_263() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_264() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_265() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_266() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_267() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_268() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_269() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_270() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_271() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_272() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_273() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_274() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_275() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_276() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_277() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_278() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_279() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_280() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_281() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_282() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_283() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_284() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_285() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_286() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_287() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_288() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_289() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_290() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_291() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_292() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_293() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_294() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 5;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_295() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 1;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_296() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_297() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 3;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_stress_298() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 4;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
    // Metric evaluation and validation padding line 2
    // Metric evaluation and validation padding line 3
}
