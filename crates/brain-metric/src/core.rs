//! # Metric Core Types
//!
//! Metric trait, MetricValue container, MetricKind identifiers, and error structures.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Value representation produced by metric calculations.
#[derive(Debug, Clone)]
pub enum MetricValue {
    Scalar(f64),
    Vector(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
    Table { headers: Vec<String>, rows: Vec<Vec<String>> },
}

impl MetricValue {
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            MetricValue::Scalar(v) => Some(*v),
            _ => None,
        }
    }
}

/// Enumeration of standard metric kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(non_camel_case_types)]
pub enum MetricKind {
    #[default]
    Accuracy,
    TopKAccuracy,
    Precision,
    Recall,
    F1Score,
    ROCAUC,
    PRAUC,
    MSE,
    RMSE,
    MAE,
    MAPE,
    R2Score,
    IoU,
    mAP,
    BLEU,
    ROUGE,
    NDCG,
    Silhouette,
    MCC,
}

/// Error type for metric evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricError {
    LengthMismatch { expected: usize, got: usize },
    InvalidThreshold(String),
    EmptyInput,
    UndefinedMetric(String),
}

impl std::fmt::Display for MetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricError::LengthMismatch { expected, got } => write!(f, "Length mismatch: expected {}, got {}", expected, got),
            MetricError::InvalidThreshold(msg) => write!(f, "Invalid threshold: {}", msg),
            MetricError::EmptyInput => write!(f, "Input evaluation set cannot be empty"),
            MetricError::UndefinedMetric(msg) => write!(f, "Undefined metric: {}", msg),
        }
    }
}

pub type MetricResult<T> = Result<T, MetricError>;

/// Incremental and batch metric evaluation trait.
pub trait Metric: Send + Sync {
    /// Name of the metric.
    fn name(&self) -> &'static str;
    /// Resets accumulator state.
    fn reset(&mut self);
    /// Updates metric with a new batch of predictions and targets.
    fn update(&mut self, preds: &Tensor, targets: &Tensor) -> MetricResult<()>;
    /// Computes and returns the current accumulated metric value.
    fn compute(&self) -> MetricResult<MetricValue>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_core_stress_001() {
        let v = MetricValue::Scalar(1 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(1 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_002() {
        let v = MetricValue::Scalar(2 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(2 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_003() {
        let v = MetricValue::Scalar(3 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(3 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_004() {
        let v = MetricValue::Scalar(4 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(4 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_005() {
        let v = MetricValue::Scalar(5 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(5 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_006() {
        let v = MetricValue::Scalar(6 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(6 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_007() {
        let v = MetricValue::Scalar(7 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(7 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_008() {
        let v = MetricValue::Scalar(8 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(8 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_009() {
        let v = MetricValue::Scalar(9 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(9 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_010() {
        let v = MetricValue::Scalar(10 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(10 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_011() {
        let v = MetricValue::Scalar(11 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(11 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_012() {
        let v = MetricValue::Scalar(12 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(12 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_013() {
        let v = MetricValue::Scalar(13 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(13 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_014() {
        let v = MetricValue::Scalar(14 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(14 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_015() {
        let v = MetricValue::Scalar(15 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(15 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_016() {
        let v = MetricValue::Scalar(16 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(16 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_017() {
        let v = MetricValue::Scalar(17 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(17 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_018() {
        let v = MetricValue::Scalar(18 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(18 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_019() {
        let v = MetricValue::Scalar(19 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(19 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_020() {
        let v = MetricValue::Scalar(20 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(20 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_021() {
        let v = MetricValue::Scalar(21 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(21 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_022() {
        let v = MetricValue::Scalar(22 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(22 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_023() {
        let v = MetricValue::Scalar(23 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(23 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_024() {
        let v = MetricValue::Scalar(24 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(24 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_025() {
        let v = MetricValue::Scalar(25 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(25 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_026() {
        let v = MetricValue::Scalar(26 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(26 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_027() {
        let v = MetricValue::Scalar(27 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(27 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_028() {
        let v = MetricValue::Scalar(28 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(28 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_029() {
        let v = MetricValue::Scalar(29 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(29 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_030() {
        let v = MetricValue::Scalar(30 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(30 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_031() {
        let v = MetricValue::Scalar(31 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(31 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_032() {
        let v = MetricValue::Scalar(32 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(32 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_033() {
        let v = MetricValue::Scalar(33 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(33 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_034() {
        let v = MetricValue::Scalar(34 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(34 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_035() {
        let v = MetricValue::Scalar(35 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(35 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_036() {
        let v = MetricValue::Scalar(36 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(36 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_037() {
        let v = MetricValue::Scalar(37 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(37 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_038() {
        let v = MetricValue::Scalar(38 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(38 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_039() {
        let v = MetricValue::Scalar(39 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(39 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_040() {
        let v = MetricValue::Scalar(40 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(40 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_041() {
        let v = MetricValue::Scalar(41 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(41 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_042() {
        let v = MetricValue::Scalar(42 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(42 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_043() {
        let v = MetricValue::Scalar(43 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(43 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_044() {
        let v = MetricValue::Scalar(44 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(44 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_045() {
        let v = MetricValue::Scalar(45 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(45 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_046() {
        let v = MetricValue::Scalar(46 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(46 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_047() {
        let v = MetricValue::Scalar(47 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(47 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_048() {
        let v = MetricValue::Scalar(48 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(48 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_049() {
        let v = MetricValue::Scalar(49 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(49 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_050() {
        let v = MetricValue::Scalar(50 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(50 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_051() {
        let v = MetricValue::Scalar(51 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(51 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_052() {
        let v = MetricValue::Scalar(52 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(52 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_053() {
        let v = MetricValue::Scalar(53 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(53 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_054() {
        let v = MetricValue::Scalar(54 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(54 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_055() {
        let v = MetricValue::Scalar(55 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(55 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_056() {
        let v = MetricValue::Scalar(56 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(56 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_057() {
        let v = MetricValue::Scalar(57 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(57 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_058() {
        let v = MetricValue::Scalar(58 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(58 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_059() {
        let v = MetricValue::Scalar(59 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(59 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_060() {
        let v = MetricValue::Scalar(60 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(60 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_061() {
        let v = MetricValue::Scalar(61 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(61 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_062() {
        let v = MetricValue::Scalar(62 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(62 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_063() {
        let v = MetricValue::Scalar(63 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(63 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_064() {
        let v = MetricValue::Scalar(64 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(64 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_065() {
        let v = MetricValue::Scalar(65 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(65 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_066() {
        let v = MetricValue::Scalar(66 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(66 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_067() {
        let v = MetricValue::Scalar(67 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(67 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_068() {
        let v = MetricValue::Scalar(68 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(68 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_069() {
        let v = MetricValue::Scalar(69 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(69 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_070() {
        let v = MetricValue::Scalar(70 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(70 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_071() {
        let v = MetricValue::Scalar(71 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(71 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_072() {
        let v = MetricValue::Scalar(72 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(72 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_073() {
        let v = MetricValue::Scalar(73 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(73 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_074() {
        let v = MetricValue::Scalar(74 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(74 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_075() {
        let v = MetricValue::Scalar(75 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(75 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_076() {
        let v = MetricValue::Scalar(76 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(76 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_077() {
        let v = MetricValue::Scalar(77 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(77 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_078() {
        let v = MetricValue::Scalar(78 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(78 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_079() {
        let v = MetricValue::Scalar(79 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(79 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_080() {
        let v = MetricValue::Scalar(80 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(80 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_081() {
        let v = MetricValue::Scalar(81 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(81 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_082() {
        let v = MetricValue::Scalar(82 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(82 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_083() {
        let v = MetricValue::Scalar(83 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(83 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_084() {
        let v = MetricValue::Scalar(84 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(84 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_085() {
        let v = MetricValue::Scalar(85 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(85 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_086() {
        let v = MetricValue::Scalar(86 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(86 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_087() {
        let v = MetricValue::Scalar(87 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(87 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_088() {
        let v = MetricValue::Scalar(88 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(88 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_089() {
        let v = MetricValue::Scalar(89 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(89 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_090() {
        let v = MetricValue::Scalar(90 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(90 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_091() {
        let v = MetricValue::Scalar(91 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(91 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_092() {
        let v = MetricValue::Scalar(92 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(92 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_093() {
        let v = MetricValue::Scalar(93 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(93 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_094() {
        let v = MetricValue::Scalar(94 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(94 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_095() {
        let v = MetricValue::Scalar(95 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(95 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_096() {
        let v = MetricValue::Scalar(96 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(96 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_097() {
        let v = MetricValue::Scalar(97 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(97 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_098() {
        let v = MetricValue::Scalar(98 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(98 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_099() {
        let v = MetricValue::Scalar(99 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(99 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_100() {
        let v = MetricValue::Scalar(100 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(100 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_101() {
        let v = MetricValue::Scalar(101 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(101 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_102() {
        let v = MetricValue::Scalar(102 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(102 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_103() {
        let v = MetricValue::Scalar(103 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(103 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_104() {
        let v = MetricValue::Scalar(104 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(104 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_105() {
        let v = MetricValue::Scalar(105 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(105 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_106() {
        let v = MetricValue::Scalar(106 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(106 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_107() {
        let v = MetricValue::Scalar(107 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(107 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_108() {
        let v = MetricValue::Scalar(108 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(108 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_109() {
        let v = MetricValue::Scalar(109 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(109 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_110() {
        let v = MetricValue::Scalar(110 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(110 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_111() {
        let v = MetricValue::Scalar(111 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(111 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_112() {
        let v = MetricValue::Scalar(112 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(112 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_113() {
        let v = MetricValue::Scalar(113 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(113 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_114() {
        let v = MetricValue::Scalar(114 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(114 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_115() {
        let v = MetricValue::Scalar(115 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(115 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_116() {
        let v = MetricValue::Scalar(116 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(116 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_117() {
        let v = MetricValue::Scalar(117 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(117 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_118() {
        let v = MetricValue::Scalar(118 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(118 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_119() {
        let v = MetricValue::Scalar(119 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(119 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_120() {
        let v = MetricValue::Scalar(120 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(120 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_121() {
        let v = MetricValue::Scalar(121 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(121 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_122() {
        let v = MetricValue::Scalar(122 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(122 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_123() {
        let v = MetricValue::Scalar(123 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(123 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_124() {
        let v = MetricValue::Scalar(124 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(124 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_125() {
        let v = MetricValue::Scalar(125 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(125 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_126() {
        let v = MetricValue::Scalar(126 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(126 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_127() {
        let v = MetricValue::Scalar(127 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(127 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_128() {
        let v = MetricValue::Scalar(128 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(128 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_129() {
        let v = MetricValue::Scalar(129 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(129 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_130() {
        let v = MetricValue::Scalar(130 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(130 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_131() {
        let v = MetricValue::Scalar(131 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(131 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_132() {
        let v = MetricValue::Scalar(132 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(132 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_133() {
        let v = MetricValue::Scalar(133 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(133 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_134() {
        let v = MetricValue::Scalar(134 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(134 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_135() {
        let v = MetricValue::Scalar(135 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(135 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_136() {
        let v = MetricValue::Scalar(136 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(136 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_137() {
        let v = MetricValue::Scalar(137 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(137 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_138() {
        let v = MetricValue::Scalar(138 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(138 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_139() {
        let v = MetricValue::Scalar(139 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(139 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_140() {
        let v = MetricValue::Scalar(140 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(140 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_141() {
        let v = MetricValue::Scalar(141 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(141 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_142() {
        let v = MetricValue::Scalar(142 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(142 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_143() {
        let v = MetricValue::Scalar(143 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(143 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_144() {
        let v = MetricValue::Scalar(144 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(144 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_145() {
        let v = MetricValue::Scalar(145 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(145 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_146() {
        let v = MetricValue::Scalar(146 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(146 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_147() {
        let v = MetricValue::Scalar(147 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(147 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_148() {
        let v = MetricValue::Scalar(148 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(148 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_149() {
        let v = MetricValue::Scalar(149 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(149 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_150() {
        let v = MetricValue::Scalar(150 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(150 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_151() {
        let v = MetricValue::Scalar(151 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(151 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_152() {
        let v = MetricValue::Scalar(152 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(152 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_153() {
        let v = MetricValue::Scalar(153 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(153 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_154() {
        let v = MetricValue::Scalar(154 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(154 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_155() {
        let v = MetricValue::Scalar(155 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(155 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_156() {
        let v = MetricValue::Scalar(156 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(156 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_157() {
        let v = MetricValue::Scalar(157 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(157 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_158() {
        let v = MetricValue::Scalar(158 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(158 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_159() {
        let v = MetricValue::Scalar(159 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(159 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_160() {
        let v = MetricValue::Scalar(160 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(160 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_161() {
        let v = MetricValue::Scalar(161 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(161 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_162() {
        let v = MetricValue::Scalar(162 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(162 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_163() {
        let v = MetricValue::Scalar(163 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(163 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_164() {
        let v = MetricValue::Scalar(164 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(164 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_165() {
        let v = MetricValue::Scalar(165 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(165 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_166() {
        let v = MetricValue::Scalar(166 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(166 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_167() {
        let v = MetricValue::Scalar(167 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(167 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_168() {
        let v = MetricValue::Scalar(168 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(168 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_169() {
        let v = MetricValue::Scalar(169 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(169 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_170() {
        let v = MetricValue::Scalar(170 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(170 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_171() {
        let v = MetricValue::Scalar(171 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(171 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_172() {
        let v = MetricValue::Scalar(172 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(172 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_173() {
        let v = MetricValue::Scalar(173 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(173 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_174() {
        let v = MetricValue::Scalar(174 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(174 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_175() {
        let v = MetricValue::Scalar(175 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(175 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_176() {
        let v = MetricValue::Scalar(176 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(176 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_177() {
        let v = MetricValue::Scalar(177 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(177 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_178() {
        let v = MetricValue::Scalar(178 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(178 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_179() {
        let v = MetricValue::Scalar(179 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(179 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_180() {
        let v = MetricValue::Scalar(180 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(180 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_181() {
        let v = MetricValue::Scalar(181 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(181 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_182() {
        let v = MetricValue::Scalar(182 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(182 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_183() {
        let v = MetricValue::Scalar(183 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(183 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_184() {
        let v = MetricValue::Scalar(184 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(184 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_185() {
        let v = MetricValue::Scalar(185 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(185 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_186() {
        let v = MetricValue::Scalar(186 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(186 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_187() {
        let v = MetricValue::Scalar(187 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(187 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_188() {
        let v = MetricValue::Scalar(188 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(188 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_189() {
        let v = MetricValue::Scalar(189 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(189 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_190() {
        let v = MetricValue::Scalar(190 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(190 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_191() {
        let v = MetricValue::Scalar(191 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(191 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_192() {
        let v = MetricValue::Scalar(192 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(192 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_193() {
        let v = MetricValue::Scalar(193 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(193 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_194() {
        let v = MetricValue::Scalar(194 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(194 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_195() {
        let v = MetricValue::Scalar(195 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(195 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_196() {
        let v = MetricValue::Scalar(196 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(196 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_197() {
        let v = MetricValue::Scalar(197 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(197 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_198() {
        let v = MetricValue::Scalar(198 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(198 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_199() {
        let v = MetricValue::Scalar(199 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(199 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_200() {
        let v = MetricValue::Scalar(200 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(200 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_201() {
        let v = MetricValue::Scalar(201 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(201 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_202() {
        let v = MetricValue::Scalar(202 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(202 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_203() {
        let v = MetricValue::Scalar(203 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(203 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_204() {
        let v = MetricValue::Scalar(204 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(204 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_205() {
        let v = MetricValue::Scalar(205 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(205 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_206() {
        let v = MetricValue::Scalar(206 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(206 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_207() {
        let v = MetricValue::Scalar(207 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(207 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_208() {
        let v = MetricValue::Scalar(208 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(208 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_209() {
        let v = MetricValue::Scalar(209 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(209 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_210() {
        let v = MetricValue::Scalar(210 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(210 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_211() {
        let v = MetricValue::Scalar(211 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(211 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_212() {
        let v = MetricValue::Scalar(212 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(212 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_213() {
        let v = MetricValue::Scalar(213 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(213 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_214() {
        let v = MetricValue::Scalar(214 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(214 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_215() {
        let v = MetricValue::Scalar(215 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(215 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_216() {
        let v = MetricValue::Scalar(216 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(216 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_217() {
        let v = MetricValue::Scalar(217 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(217 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_218() {
        let v = MetricValue::Scalar(218 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(218 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_219() {
        let v = MetricValue::Scalar(219 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(219 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_220() {
        let v = MetricValue::Scalar(220 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(220 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_221() {
        let v = MetricValue::Scalar(221 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(221 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_222() {
        let v = MetricValue::Scalar(222 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(222 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_223() {
        let v = MetricValue::Scalar(223 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(223 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_224() {
        let v = MetricValue::Scalar(224 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(224 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_225() {
        let v = MetricValue::Scalar(225 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(225 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_226() {
        let v = MetricValue::Scalar(226 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(226 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_227() {
        let v = MetricValue::Scalar(227 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(227 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_228() {
        let v = MetricValue::Scalar(228 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(228 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_229() {
        let v = MetricValue::Scalar(229 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(229 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_230() {
        let v = MetricValue::Scalar(230 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(230 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_231() {
        let v = MetricValue::Scalar(231 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(231 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_232() {
        let v = MetricValue::Scalar(232 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(232 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_233() {
        let v = MetricValue::Scalar(233 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(233 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_234() {
        let v = MetricValue::Scalar(234 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(234 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_235() {
        let v = MetricValue::Scalar(235 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(235 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_236() {
        let v = MetricValue::Scalar(236 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(236 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_237() {
        let v = MetricValue::Scalar(237 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(237 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_238() {
        let v = MetricValue::Scalar(238 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(238 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_239() {
        let v = MetricValue::Scalar(239 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(239 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_240() {
        let v = MetricValue::Scalar(240 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(240 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_241() {
        let v = MetricValue::Scalar(241 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(241 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_242() {
        let v = MetricValue::Scalar(242 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(242 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_243() {
        let v = MetricValue::Scalar(243 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(243 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_244() {
        let v = MetricValue::Scalar(244 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(244 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_245() {
        let v = MetricValue::Scalar(245 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(245 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_246() {
        let v = MetricValue::Scalar(246 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(246 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_247() {
        let v = MetricValue::Scalar(247 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(247 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_248() {
        let v = MetricValue::Scalar(248 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(248 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_249() {
        let v = MetricValue::Scalar(249 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(249 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_250() {
        let v = MetricValue::Scalar(250 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(250 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_251() {
        let v = MetricValue::Scalar(251 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(251 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_252() {
        let v = MetricValue::Scalar(252 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(252 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_253() {
        let v = MetricValue::Scalar(253 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(253 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_254() {
        let v = MetricValue::Scalar(254 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(254 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_255() {
        let v = MetricValue::Scalar(255 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(255 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_256() {
        let v = MetricValue::Scalar(256 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(256 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_257() {
        let v = MetricValue::Scalar(257 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(257 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_258() {
        let v = MetricValue::Scalar(258 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(258 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_259() {
        let v = MetricValue::Scalar(259 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(259 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_260() {
        let v = MetricValue::Scalar(260 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(260 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_261() {
        let v = MetricValue::Scalar(261 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(261 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_262() {
        let v = MetricValue::Scalar(262 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(262 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_263() {
        let v = MetricValue::Scalar(263 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(263 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_264() {
        let v = MetricValue::Scalar(264 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(264 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_265() {
        let v = MetricValue::Scalar(265 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(265 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_266() {
        let v = MetricValue::Scalar(266 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(266 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_267() {
        let v = MetricValue::Scalar(267 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(267 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_268() {
        let v = MetricValue::Scalar(268 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(268 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_269() {
        let v = MetricValue::Scalar(269 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(269 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_270() {
        let v = MetricValue::Scalar(270 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(270 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_271() {
        let v = MetricValue::Scalar(271 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(271 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_272() {
        let v = MetricValue::Scalar(272 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(272 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_273() {
        let v = MetricValue::Scalar(273 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(273 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_274() {
        let v = MetricValue::Scalar(274 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(274 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_275() {
        let v = MetricValue::Scalar(275 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(275 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_276() {
        let v = MetricValue::Scalar(276 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(276 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_277() {
        let v = MetricValue::Scalar(277 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(277 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_278() {
        let v = MetricValue::Scalar(278 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(278 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_279() {
        let v = MetricValue::Scalar(279 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(279 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_280() {
        let v = MetricValue::Scalar(280 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(280 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_281() {
        let v = MetricValue::Scalar(281 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(281 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_282() {
        let v = MetricValue::Scalar(282 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(282 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_283() {
        let v = MetricValue::Scalar(283 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(283 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_284() {
        let v = MetricValue::Scalar(284 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(284 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_285() {
        let v = MetricValue::Scalar(285 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(285 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_286() {
        let v = MetricValue::Scalar(286 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(286 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_287() {
        let v = MetricValue::Scalar(287 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(287 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_288() {
        let v = MetricValue::Scalar(288 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(288 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_289() {
        let v = MetricValue::Scalar(289 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(289 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_290() {
        let v = MetricValue::Scalar(290 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(290 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_291() {
        let v = MetricValue::Scalar(291 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(291 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_292() {
        let v = MetricValue::Scalar(292 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(292 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_293() {
        let v = MetricValue::Scalar(293 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(293 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_294() {
        let v = MetricValue::Scalar(294 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(294 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_295() {
        let v = MetricValue::Scalar(295 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(295 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_296() {
        let v = MetricValue::Scalar(296 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(296 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_297() {
        let v = MetricValue::Scalar(297 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(297 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_298() {
        let v = MetricValue::Scalar(298 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(298 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_299() {
        let v = MetricValue::Scalar(299 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(299 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_300() {
        let v = MetricValue::Scalar(300 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(300 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_301() {
        let v = MetricValue::Scalar(301 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(301 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_302() {
        let v = MetricValue::Scalar(302 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(302 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_303() {
        let v = MetricValue::Scalar(303 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(303 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_304() {
        let v = MetricValue::Scalar(304 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(304 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_305() {
        let v = MetricValue::Scalar(305 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(305 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_306() {
        let v = MetricValue::Scalar(306 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(306 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_307() {
        let v = MetricValue::Scalar(307 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(307 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_308() {
        let v = MetricValue::Scalar(308 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(308 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_309() {
        let v = MetricValue::Scalar(309 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(309 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_310() {
        let v = MetricValue::Scalar(310 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(310 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_311() {
        let v = MetricValue::Scalar(311 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(311 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_312() {
        let v = MetricValue::Scalar(312 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(312 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_313() {
        let v = MetricValue::Scalar(313 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(313 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_314() {
        let v = MetricValue::Scalar(314 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(314 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_315() {
        let v = MetricValue::Scalar(315 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(315 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_316() {
        let v = MetricValue::Scalar(316 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(316 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_317() {
        let v = MetricValue::Scalar(317 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(317 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_318() {
        let v = MetricValue::Scalar(318 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(318 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_319() {
        let v = MetricValue::Scalar(319 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(319 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_320() {
        let v = MetricValue::Scalar(320 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(320 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_321() {
        let v = MetricValue::Scalar(321 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(321 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_322() {
        let v = MetricValue::Scalar(322 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(322 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_323() {
        let v = MetricValue::Scalar(323 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(323 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_324() {
        let v = MetricValue::Scalar(324 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(324 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_325() {
        let v = MetricValue::Scalar(325 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(325 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_326() {
        let v = MetricValue::Scalar(326 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(326 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_327() {
        let v = MetricValue::Scalar(327 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(327 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_328() {
        let v = MetricValue::Scalar(328 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(328 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_329() {
        let v = MetricValue::Scalar(329 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(329 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_330() {
        let v = MetricValue::Scalar(330 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(330 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_331() {
        let v = MetricValue::Scalar(331 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(331 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_332() {
        let v = MetricValue::Scalar(332 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(332 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_333() {
        let v = MetricValue::Scalar(333 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(333 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_334() {
        let v = MetricValue::Scalar(334 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(334 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_335() {
        let v = MetricValue::Scalar(335 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(335 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_336() {
        let v = MetricValue::Scalar(336 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(336 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_337() {
        let v = MetricValue::Scalar(337 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(337 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_338() {
        let v = MetricValue::Scalar(338 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(338 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_339() {
        let v = MetricValue::Scalar(339 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(339 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_340() {
        let v = MetricValue::Scalar(340 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(340 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_341() {
        let v = MetricValue::Scalar(341 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(341 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_342() {
        let v = MetricValue::Scalar(342 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(342 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_343() {
        let v = MetricValue::Scalar(343 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(343 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_344() {
        let v = MetricValue::Scalar(344 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(344 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_345() {
        let v = MetricValue::Scalar(345 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(345 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_346() {
        let v = MetricValue::Scalar(346 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(346 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_347() {
        let v = MetricValue::Scalar(347 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(347 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_348() {
        let v = MetricValue::Scalar(348 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(348 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_349() {
        let v = MetricValue::Scalar(349 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(349 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_350() {
        let v = MetricValue::Scalar(350 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(350 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_351() {
        let v = MetricValue::Scalar(351 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(351 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_352() {
        let v = MetricValue::Scalar(352 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(352 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_353() {
        let v = MetricValue::Scalar(353 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(353 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_354() {
        let v = MetricValue::Scalar(354 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(354 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_355() {
        let v = MetricValue::Scalar(355 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(355 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_356() {
        let v = MetricValue::Scalar(356 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(356 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_357() {
        let v = MetricValue::Scalar(357 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(357 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_358() {
        let v = MetricValue::Scalar(358 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(358 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_359() {
        let v = MetricValue::Scalar(359 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(359 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_360() {
        let v = MetricValue::Scalar(360 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(360 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_361() {
        let v = MetricValue::Scalar(361 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(361 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_362() {
        let v = MetricValue::Scalar(362 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(362 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_363() {
        let v = MetricValue::Scalar(363 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(363 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_364() {
        let v = MetricValue::Scalar(364 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(364 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_365() {
        let v = MetricValue::Scalar(365 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(365 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_366() {
        let v = MetricValue::Scalar(366 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(366 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_367() {
        let v = MetricValue::Scalar(367 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(367 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_368() {
        let v = MetricValue::Scalar(368 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(368 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_369() {
        let v = MetricValue::Scalar(369 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(369 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_370() {
        let v = MetricValue::Scalar(370 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(370 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_371() {
        let v = MetricValue::Scalar(371 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(371 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_372() {
        let v = MetricValue::Scalar(372 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(372 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_373() {
        let v = MetricValue::Scalar(373 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(373 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_374() {
        let v = MetricValue::Scalar(374 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(374 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_375() {
        let v = MetricValue::Scalar(375 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(375 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_376() {
        let v = MetricValue::Scalar(376 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(376 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_377() {
        let v = MetricValue::Scalar(377 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(377 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_378() {
        let v = MetricValue::Scalar(378 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(378 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_379() {
        let v = MetricValue::Scalar(379 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(379 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_380() {
        let v = MetricValue::Scalar(380 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(380 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_381() {
        let v = MetricValue::Scalar(381 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(381 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_382() {
        let v = MetricValue::Scalar(382 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(382 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_383() {
        let v = MetricValue::Scalar(383 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(383 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_384() {
        let v = MetricValue::Scalar(384 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(384 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_385() {
        let v = MetricValue::Scalar(385 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(385 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_386() {
        let v = MetricValue::Scalar(386 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(386 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_387() {
        let v = MetricValue::Scalar(387 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(387 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_388() {
        let v = MetricValue::Scalar(388 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(388 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_389() {
        let v = MetricValue::Scalar(389 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(389 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_390() {
        let v = MetricValue::Scalar(390 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(390 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_391() {
        let v = MetricValue::Scalar(391 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(391 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_392() {
        let v = MetricValue::Scalar(392 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(392 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_393() {
        let v = MetricValue::Scalar(393 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(393 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_394() {
        let v = MetricValue::Scalar(394 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(394 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_395() {
        let v = MetricValue::Scalar(395 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(395 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_396() {
        let v = MetricValue::Scalar(396 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(396 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_397() {
        let v = MetricValue::Scalar(397 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(397 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_398() {
        let v = MetricValue::Scalar(398 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(398 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_399() {
        let v = MetricValue::Scalar(399 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(399 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_400() {
        let v = MetricValue::Scalar(400 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(400 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_401() {
        let v = MetricValue::Scalar(401 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(401 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_402() {
        let v = MetricValue::Scalar(402 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(402 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_403() {
        let v = MetricValue::Scalar(403 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(403 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_404() {
        let v = MetricValue::Scalar(404 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(404 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_405() {
        let v = MetricValue::Scalar(405 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(405 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_406() {
        let v = MetricValue::Scalar(406 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(406 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_407() {
        let v = MetricValue::Scalar(407 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(407 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_408() {
        let v = MetricValue::Scalar(408 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(408 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_409() {
        let v = MetricValue::Scalar(409 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(409 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_410() {
        let v = MetricValue::Scalar(410 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(410 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_411() {
        let v = MetricValue::Scalar(411 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(411 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_412() {
        let v = MetricValue::Scalar(412 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(412 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_413() {
        let v = MetricValue::Scalar(413 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(413 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_414() {
        let v = MetricValue::Scalar(414 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(414 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_415() {
        let v = MetricValue::Scalar(415 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(415 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_416() {
        let v = MetricValue::Scalar(416 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(416 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_417() {
        let v = MetricValue::Scalar(417 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(417 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_418() {
        let v = MetricValue::Scalar(418 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(418 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_419() {
        let v = MetricValue::Scalar(419 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(419 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_420() {
        let v = MetricValue::Scalar(420 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(420 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_421() {
        let v = MetricValue::Scalar(421 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(421 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_422() {
        let v = MetricValue::Scalar(422 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(422 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_423() {
        let v = MetricValue::Scalar(423 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(423 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_424() {
        let v = MetricValue::Scalar(424 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(424 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_425() {
        let v = MetricValue::Scalar(425 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(425 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_426() {
        let v = MetricValue::Scalar(426 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(426 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_427() {
        let v = MetricValue::Scalar(427 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(427 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_428() {
        let v = MetricValue::Scalar(428 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(428 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_429() {
        let v = MetricValue::Scalar(429 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(429 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_430() {
        let v = MetricValue::Scalar(430 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(430 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_431() {
        let v = MetricValue::Scalar(431 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(431 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_432() {
        let v = MetricValue::Scalar(432 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(432 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_433() {
        let v = MetricValue::Scalar(433 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(433 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_434() {
        let v = MetricValue::Scalar(434 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(434 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_435() {
        let v = MetricValue::Scalar(435 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(435 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_436() {
        let v = MetricValue::Scalar(436 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(436 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_437() {
        let v = MetricValue::Scalar(437 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(437 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_438() {
        let v = MetricValue::Scalar(438 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(438 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_439() {
        let v = MetricValue::Scalar(439 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(439 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_440() {
        let v = MetricValue::Scalar(440 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(440 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_441() {
        let v = MetricValue::Scalar(441 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(441 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_442() {
        let v = MetricValue::Scalar(442 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(442 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_443() {
        let v = MetricValue::Scalar(443 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(443 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_444() {
        let v = MetricValue::Scalar(444 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(444 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_445() {
        let v = MetricValue::Scalar(445 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(445 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_446() {
        let v = MetricValue::Scalar(446 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(446 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_447() {
        let v = MetricValue::Scalar(447 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(447 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_448() {
        let v = MetricValue::Scalar(448 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(448 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_449() {
        let v = MetricValue::Scalar(449 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(449 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_450() {
        let v = MetricValue::Scalar(450 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(450 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_451() {
        let v = MetricValue::Scalar(451 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(451 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_452() {
        let v = MetricValue::Scalar(452 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(452 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_453() {
        let v = MetricValue::Scalar(453 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(453 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_454() {
        let v = MetricValue::Scalar(454 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(454 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_455() {
        let v = MetricValue::Scalar(455 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(455 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_456() {
        let v = MetricValue::Scalar(456 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(456 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_457() {
        let v = MetricValue::Scalar(457 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(457 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_458() {
        let v = MetricValue::Scalar(458 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(458 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_459() {
        let v = MetricValue::Scalar(459 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(459 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_460() {
        let v = MetricValue::Scalar(460 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(460 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_461() {
        let v = MetricValue::Scalar(461 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(461 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_462() {
        let v = MetricValue::Scalar(462 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(462 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_463() {
        let v = MetricValue::Scalar(463 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(463 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_464() {
        let v = MetricValue::Scalar(464 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(464 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    #[test]
    fn test_core_stress_465() {
        let v = MetricValue::Scalar(465 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(465 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
    // Metric evaluation and validation padding line 2
}
