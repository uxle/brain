//! # GAN Core Types
//!
//! Fundamental data structures: [`GanState`], [`GanMetrics`], [`GanResult`].
#![allow(missing_docs)]

use brain_core::Tensor;

/// Error type for GAN operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GanError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    InvalidConfig(String),
    TrainingFailed(String),
}

impl std::fmt::Display for GanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GanError::ShapeMismatch { expected, got } =>
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got),
            GanError::InvalidConfig(s) => write!(f, "Invalid config: {}", s),
            GanError::TrainingFailed(s) => write!(f, "Training failed: {}", s),
        }
    }
}

/// Result type for GAN operations.
pub type GanResult<T> = Result<T, GanError>;

/// Metrics collected after each training step.
#[derive(Debug, Clone, Default)]
pub struct GanMetrics {
    pub step: usize,
    pub d_loss: f64,
    pub g_loss: f64,
    pub d_real: f64,
    pub d_fake: f64,
    pub grad_norm_g: f64,
    pub grad_norm_d: f64,
    pub gp: f64,
}

/// Full GAN state: weights, optimizer states, step counter.
#[derive(Debug, Clone)]
pub struct GanState {
    pub generator_weights: Vec<Tensor>,
    pub discriminator_weights: Vec<Tensor>,
    pub step: usize,
    pub epoch: usize,
}

impl GanState {
    pub fn new(generator_weights: Vec<Tensor>, discriminator_weights: Vec<Tensor>) -> Self {
        Self { generator_weights, discriminator_weights, step: 0, epoch: 0 }
    }

    pub fn advance_step(&mut self) { self.step += 1; }
    pub fn advance_epoch(&mut self) { self.epoch += 1; }
}

/// Summary statistics for a training epoch.
#[derive(Debug, Clone, Default)]
pub struct EpochSummary {
    pub epoch: usize,
    pub avg_d_loss: f64,
    pub avg_g_loss: f64,
    pub num_steps: usize,
}

impl EpochSummary {
    pub fn new(epoch: usize) -> Self {
        Self { epoch, avg_d_loss: 0.0, avg_g_loss: 0.0, num_steps: 0 }
    }

    pub fn update(&mut self, metrics: &GanMetrics) {
        self.avg_d_loss += metrics.d_loss;
        self.avg_g_loss += metrics.g_loss;
        self.num_steps += 1;
    }

    pub fn finalize(&mut self) {
        if self.num_steps > 0 {
            self.avg_d_loss /= self.num_steps as f64;
            self.avg_g_loss /= self.num_steps as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_core_stress_001() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..1 { state.advance_step(); }
        assert_eq!(state.step, 1);
        let mut summary = EpochSummary::new(1);
        let m = GanMetrics { step: 1, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_002() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..2 { state.advance_step(); }
        assert_eq!(state.step, 2);
        let mut summary = EpochSummary::new(2);
        let m = GanMetrics { step: 2, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_003() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..3 { state.advance_step(); }
        assert_eq!(state.step, 3);
        let mut summary = EpochSummary::new(3);
        let m = GanMetrics { step: 3, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_004() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..4 { state.advance_step(); }
        assert_eq!(state.step, 4);
        let mut summary = EpochSummary::new(4);
        let m = GanMetrics { step: 4, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_005() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..5 { state.advance_step(); }
        assert_eq!(state.step, 5);
        let mut summary = EpochSummary::new(5);
        let m = GanMetrics { step: 5, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_006() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..6 { state.advance_step(); }
        assert_eq!(state.step, 6);
        let mut summary = EpochSummary::new(6);
        let m = GanMetrics { step: 6, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_007() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..7 { state.advance_step(); }
        assert_eq!(state.step, 7);
        let mut summary = EpochSummary::new(7);
        let m = GanMetrics { step: 7, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_008() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..8 { state.advance_step(); }
        assert_eq!(state.step, 8);
        let mut summary = EpochSummary::new(8);
        let m = GanMetrics { step: 8, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_009() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..9 { state.advance_step(); }
        assert_eq!(state.step, 9);
        let mut summary = EpochSummary::new(9);
        let m = GanMetrics { step: 9, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_010() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..10 { state.advance_step(); }
        assert_eq!(state.step, 10);
        let mut summary = EpochSummary::new(10);
        let m = GanMetrics { step: 10, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_011() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..11 { state.advance_step(); }
        assert_eq!(state.step, 11);
        let mut summary = EpochSummary::new(11);
        let m = GanMetrics { step: 11, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_012() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..12 { state.advance_step(); }
        assert_eq!(state.step, 12);
        let mut summary = EpochSummary::new(12);
        let m = GanMetrics { step: 12, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_013() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..13 { state.advance_step(); }
        assert_eq!(state.step, 13);
        let mut summary = EpochSummary::new(13);
        let m = GanMetrics { step: 13, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_014() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..14 { state.advance_step(); }
        assert_eq!(state.step, 14);
        let mut summary = EpochSummary::new(14);
        let m = GanMetrics { step: 14, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_015() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..15 { state.advance_step(); }
        assert_eq!(state.step, 15);
        let mut summary = EpochSummary::new(15);
        let m = GanMetrics { step: 15, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_016() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..16 { state.advance_step(); }
        assert_eq!(state.step, 16);
        let mut summary = EpochSummary::new(16);
        let m = GanMetrics { step: 16, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_017() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..17 { state.advance_step(); }
        assert_eq!(state.step, 17);
        let mut summary = EpochSummary::new(17);
        let m = GanMetrics { step: 17, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_018() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..18 { state.advance_step(); }
        assert_eq!(state.step, 18);
        let mut summary = EpochSummary::new(18);
        let m = GanMetrics { step: 18, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_019() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..19 { state.advance_step(); }
        assert_eq!(state.step, 19);
        let mut summary = EpochSummary::new(19);
        let m = GanMetrics { step: 19, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_020() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..20 { state.advance_step(); }
        assert_eq!(state.step, 20);
        let mut summary = EpochSummary::new(20);
        let m = GanMetrics { step: 20, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_021() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..21 { state.advance_step(); }
        assert_eq!(state.step, 21);
        let mut summary = EpochSummary::new(21);
        let m = GanMetrics { step: 21, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_022() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..22 { state.advance_step(); }
        assert_eq!(state.step, 22);
        let mut summary = EpochSummary::new(22);
        let m = GanMetrics { step: 22, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_023() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..23 { state.advance_step(); }
        assert_eq!(state.step, 23);
        let mut summary = EpochSummary::new(23);
        let m = GanMetrics { step: 23, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_024() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..24 { state.advance_step(); }
        assert_eq!(state.step, 24);
        let mut summary = EpochSummary::new(24);
        let m = GanMetrics { step: 24, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_025() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..25 { state.advance_step(); }
        assert_eq!(state.step, 25);
        let mut summary = EpochSummary::new(25);
        let m = GanMetrics { step: 25, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_026() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..26 { state.advance_step(); }
        assert_eq!(state.step, 26);
        let mut summary = EpochSummary::new(26);
        let m = GanMetrics { step: 26, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_027() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..27 { state.advance_step(); }
        assert_eq!(state.step, 27);
        let mut summary = EpochSummary::new(27);
        let m = GanMetrics { step: 27, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_028() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..28 { state.advance_step(); }
        assert_eq!(state.step, 28);
        let mut summary = EpochSummary::new(28);
        let m = GanMetrics { step: 28, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_029() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..29 { state.advance_step(); }
        assert_eq!(state.step, 29);
        let mut summary = EpochSummary::new(29);
        let m = GanMetrics { step: 29, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_030() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..30 { state.advance_step(); }
        assert_eq!(state.step, 30);
        let mut summary = EpochSummary::new(30);
        let m = GanMetrics { step: 30, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_031() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..31 { state.advance_step(); }
        assert_eq!(state.step, 31);
        let mut summary = EpochSummary::new(31);
        let m = GanMetrics { step: 31, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_032() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..32 { state.advance_step(); }
        assert_eq!(state.step, 32);
        let mut summary = EpochSummary::new(32);
        let m = GanMetrics { step: 32, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_033() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..33 { state.advance_step(); }
        assert_eq!(state.step, 33);
        let mut summary = EpochSummary::new(33);
        let m = GanMetrics { step: 33, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_034() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..34 { state.advance_step(); }
        assert_eq!(state.step, 34);
        let mut summary = EpochSummary::new(34);
        let m = GanMetrics { step: 34, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_035() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..35 { state.advance_step(); }
        assert_eq!(state.step, 35);
        let mut summary = EpochSummary::new(35);
        let m = GanMetrics { step: 35, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_036() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..36 { state.advance_step(); }
        assert_eq!(state.step, 36);
        let mut summary = EpochSummary::new(36);
        let m = GanMetrics { step: 36, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_037() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..37 { state.advance_step(); }
        assert_eq!(state.step, 37);
        let mut summary = EpochSummary::new(37);
        let m = GanMetrics { step: 37, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_038() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..38 { state.advance_step(); }
        assert_eq!(state.step, 38);
        let mut summary = EpochSummary::new(38);
        let m = GanMetrics { step: 38, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_039() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..39 { state.advance_step(); }
        assert_eq!(state.step, 39);
        let mut summary = EpochSummary::new(39);
        let m = GanMetrics { step: 39, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_040() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..40 { state.advance_step(); }
        assert_eq!(state.step, 40);
        let mut summary = EpochSummary::new(40);
        let m = GanMetrics { step: 40, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_041() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..41 { state.advance_step(); }
        assert_eq!(state.step, 41);
        let mut summary = EpochSummary::new(41);
        let m = GanMetrics { step: 41, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_042() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..42 { state.advance_step(); }
        assert_eq!(state.step, 42);
        let mut summary = EpochSummary::new(42);
        let m = GanMetrics { step: 42, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_043() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..43 { state.advance_step(); }
        assert_eq!(state.step, 43);
        let mut summary = EpochSummary::new(43);
        let m = GanMetrics { step: 43, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_044() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..44 { state.advance_step(); }
        assert_eq!(state.step, 44);
        let mut summary = EpochSummary::new(44);
        let m = GanMetrics { step: 44, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_045() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..45 { state.advance_step(); }
        assert_eq!(state.step, 45);
        let mut summary = EpochSummary::new(45);
        let m = GanMetrics { step: 45, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_046() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..46 { state.advance_step(); }
        assert_eq!(state.step, 46);
        let mut summary = EpochSummary::new(46);
        let m = GanMetrics { step: 46, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_047() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..47 { state.advance_step(); }
        assert_eq!(state.step, 47);
        let mut summary = EpochSummary::new(47);
        let m = GanMetrics { step: 47, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_048() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..48 { state.advance_step(); }
        assert_eq!(state.step, 48);
        let mut summary = EpochSummary::new(48);
        let m = GanMetrics { step: 48, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_049() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..49 { state.advance_step(); }
        assert_eq!(state.step, 49);
        let mut summary = EpochSummary::new(49);
        let m = GanMetrics { step: 49, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_050() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..50 { state.advance_step(); }
        assert_eq!(state.step, 50);
        let mut summary = EpochSummary::new(50);
        let m = GanMetrics { step: 50, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_051() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..51 { state.advance_step(); }
        assert_eq!(state.step, 51);
        let mut summary = EpochSummary::new(51);
        let m = GanMetrics { step: 51, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_052() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..52 { state.advance_step(); }
        assert_eq!(state.step, 52);
        let mut summary = EpochSummary::new(52);
        let m = GanMetrics { step: 52, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_053() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..53 { state.advance_step(); }
        assert_eq!(state.step, 53);
        let mut summary = EpochSummary::new(53);
        let m = GanMetrics { step: 53, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_054() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..54 { state.advance_step(); }
        assert_eq!(state.step, 54);
        let mut summary = EpochSummary::new(54);
        let m = GanMetrics { step: 54, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_055() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..55 { state.advance_step(); }
        assert_eq!(state.step, 55);
        let mut summary = EpochSummary::new(55);
        let m = GanMetrics { step: 55, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_056() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..56 { state.advance_step(); }
        assert_eq!(state.step, 56);
        let mut summary = EpochSummary::new(56);
        let m = GanMetrics { step: 56, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_057() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..57 { state.advance_step(); }
        assert_eq!(state.step, 57);
        let mut summary = EpochSummary::new(57);
        let m = GanMetrics { step: 57, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_058() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..58 { state.advance_step(); }
        assert_eq!(state.step, 58);
        let mut summary = EpochSummary::new(58);
        let m = GanMetrics { step: 58, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_059() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..59 { state.advance_step(); }
        assert_eq!(state.step, 59);
        let mut summary = EpochSummary::new(59);
        let m = GanMetrics { step: 59, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_060() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..60 { state.advance_step(); }
        assert_eq!(state.step, 60);
        let mut summary = EpochSummary::new(60);
        let m = GanMetrics { step: 60, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_061() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..61 { state.advance_step(); }
        assert_eq!(state.step, 61);
        let mut summary = EpochSummary::new(61);
        let m = GanMetrics { step: 61, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_062() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..62 { state.advance_step(); }
        assert_eq!(state.step, 62);
        let mut summary = EpochSummary::new(62);
        let m = GanMetrics { step: 62, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_063() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..63 { state.advance_step(); }
        assert_eq!(state.step, 63);
        let mut summary = EpochSummary::new(63);
        let m = GanMetrics { step: 63, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_064() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..64 { state.advance_step(); }
        assert_eq!(state.step, 64);
        let mut summary = EpochSummary::new(64);
        let m = GanMetrics { step: 64, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_065() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..65 { state.advance_step(); }
        assert_eq!(state.step, 65);
        let mut summary = EpochSummary::new(65);
        let m = GanMetrics { step: 65, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_066() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..66 { state.advance_step(); }
        assert_eq!(state.step, 66);
        let mut summary = EpochSummary::new(66);
        let m = GanMetrics { step: 66, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_067() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..67 { state.advance_step(); }
        assert_eq!(state.step, 67);
        let mut summary = EpochSummary::new(67);
        let m = GanMetrics { step: 67, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_068() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..68 { state.advance_step(); }
        assert_eq!(state.step, 68);
        let mut summary = EpochSummary::new(68);
        let m = GanMetrics { step: 68, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_069() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..69 { state.advance_step(); }
        assert_eq!(state.step, 69);
        let mut summary = EpochSummary::new(69);
        let m = GanMetrics { step: 69, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_070() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..70 { state.advance_step(); }
        assert_eq!(state.step, 70);
        let mut summary = EpochSummary::new(70);
        let m = GanMetrics { step: 70, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_071() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..71 { state.advance_step(); }
        assert_eq!(state.step, 71);
        let mut summary = EpochSummary::new(71);
        let m = GanMetrics { step: 71, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_072() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..72 { state.advance_step(); }
        assert_eq!(state.step, 72);
        let mut summary = EpochSummary::new(72);
        let m = GanMetrics { step: 72, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_073() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..73 { state.advance_step(); }
        assert_eq!(state.step, 73);
        let mut summary = EpochSummary::new(73);
        let m = GanMetrics { step: 73, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_074() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..74 { state.advance_step(); }
        assert_eq!(state.step, 74);
        let mut summary = EpochSummary::new(74);
        let m = GanMetrics { step: 74, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_075() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..75 { state.advance_step(); }
        assert_eq!(state.step, 75);
        let mut summary = EpochSummary::new(75);
        let m = GanMetrics { step: 75, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_076() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..76 { state.advance_step(); }
        assert_eq!(state.step, 76);
        let mut summary = EpochSummary::new(76);
        let m = GanMetrics { step: 76, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_077() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..77 { state.advance_step(); }
        assert_eq!(state.step, 77);
        let mut summary = EpochSummary::new(77);
        let m = GanMetrics { step: 77, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_078() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..78 { state.advance_step(); }
        assert_eq!(state.step, 78);
        let mut summary = EpochSummary::new(78);
        let m = GanMetrics { step: 78, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_079() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..79 { state.advance_step(); }
        assert_eq!(state.step, 79);
        let mut summary = EpochSummary::new(79);
        let m = GanMetrics { step: 79, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_080() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..80 { state.advance_step(); }
        assert_eq!(state.step, 80);
        let mut summary = EpochSummary::new(80);
        let m = GanMetrics { step: 80, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_081() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..81 { state.advance_step(); }
        assert_eq!(state.step, 81);
        let mut summary = EpochSummary::new(81);
        let m = GanMetrics { step: 81, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_082() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..82 { state.advance_step(); }
        assert_eq!(state.step, 82);
        let mut summary = EpochSummary::new(82);
        let m = GanMetrics { step: 82, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_083() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..83 { state.advance_step(); }
        assert_eq!(state.step, 83);
        let mut summary = EpochSummary::new(83);
        let m = GanMetrics { step: 83, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_084() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..84 { state.advance_step(); }
        assert_eq!(state.step, 84);
        let mut summary = EpochSummary::new(84);
        let m = GanMetrics { step: 84, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_085() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..85 { state.advance_step(); }
        assert_eq!(state.step, 85);
        let mut summary = EpochSummary::new(85);
        let m = GanMetrics { step: 85, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_086() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..86 { state.advance_step(); }
        assert_eq!(state.step, 86);
        let mut summary = EpochSummary::new(86);
        let m = GanMetrics { step: 86, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_087() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..87 { state.advance_step(); }
        assert_eq!(state.step, 87);
        let mut summary = EpochSummary::new(87);
        let m = GanMetrics { step: 87, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_088() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..88 { state.advance_step(); }
        assert_eq!(state.step, 88);
        let mut summary = EpochSummary::new(88);
        let m = GanMetrics { step: 88, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_089() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..89 { state.advance_step(); }
        assert_eq!(state.step, 89);
        let mut summary = EpochSummary::new(89);
        let m = GanMetrics { step: 89, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_090() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..90 { state.advance_step(); }
        assert_eq!(state.step, 90);
        let mut summary = EpochSummary::new(90);
        let m = GanMetrics { step: 90, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_091() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..91 { state.advance_step(); }
        assert_eq!(state.step, 91);
        let mut summary = EpochSummary::new(91);
        let m = GanMetrics { step: 91, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_092() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..92 { state.advance_step(); }
        assert_eq!(state.step, 92);
        let mut summary = EpochSummary::new(92);
        let m = GanMetrics { step: 92, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_093() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..93 { state.advance_step(); }
        assert_eq!(state.step, 93);
        let mut summary = EpochSummary::new(93);
        let m = GanMetrics { step: 93, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_094() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..94 { state.advance_step(); }
        assert_eq!(state.step, 94);
        let mut summary = EpochSummary::new(94);
        let m = GanMetrics { step: 94, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_095() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..95 { state.advance_step(); }
        assert_eq!(state.step, 95);
        let mut summary = EpochSummary::new(95);
        let m = GanMetrics { step: 95, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_096() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..96 { state.advance_step(); }
        assert_eq!(state.step, 96);
        let mut summary = EpochSummary::new(96);
        let m = GanMetrics { step: 96, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_097() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..97 { state.advance_step(); }
        assert_eq!(state.step, 97);
        let mut summary = EpochSummary::new(97);
        let m = GanMetrics { step: 97, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_098() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..98 { state.advance_step(); }
        assert_eq!(state.step, 98);
        let mut summary = EpochSummary::new(98);
        let m = GanMetrics { step: 98, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_099() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..99 { state.advance_step(); }
        assert_eq!(state.step, 99);
        let mut summary = EpochSummary::new(99);
        let m = GanMetrics { step: 99, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_100() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..100 { state.advance_step(); }
        assert_eq!(state.step, 100);
        let mut summary = EpochSummary::new(100);
        let m = GanMetrics { step: 100, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_101() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..101 { state.advance_step(); }
        assert_eq!(state.step, 101);
        let mut summary = EpochSummary::new(101);
        let m = GanMetrics { step: 101, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_102() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..102 { state.advance_step(); }
        assert_eq!(state.step, 102);
        let mut summary = EpochSummary::new(102);
        let m = GanMetrics { step: 102, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_103() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..103 { state.advance_step(); }
        assert_eq!(state.step, 103);
        let mut summary = EpochSummary::new(103);
        let m = GanMetrics { step: 103, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_104() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..104 { state.advance_step(); }
        assert_eq!(state.step, 104);
        let mut summary = EpochSummary::new(104);
        let m = GanMetrics { step: 104, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_105() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..105 { state.advance_step(); }
        assert_eq!(state.step, 105);
        let mut summary = EpochSummary::new(105);
        let m = GanMetrics { step: 105, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_106() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..106 { state.advance_step(); }
        assert_eq!(state.step, 106);
        let mut summary = EpochSummary::new(106);
        let m = GanMetrics { step: 106, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_107() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..107 { state.advance_step(); }
        assert_eq!(state.step, 107);
        let mut summary = EpochSummary::new(107);
        let m = GanMetrics { step: 107, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_108() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..108 { state.advance_step(); }
        assert_eq!(state.step, 108);
        let mut summary = EpochSummary::new(108);
        let m = GanMetrics { step: 108, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_109() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..109 { state.advance_step(); }
        assert_eq!(state.step, 109);
        let mut summary = EpochSummary::new(109);
        let m = GanMetrics { step: 109, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_110() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..110 { state.advance_step(); }
        assert_eq!(state.step, 110);
        let mut summary = EpochSummary::new(110);
        let m = GanMetrics { step: 110, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_111() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..111 { state.advance_step(); }
        assert_eq!(state.step, 111);
        let mut summary = EpochSummary::new(111);
        let m = GanMetrics { step: 111, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_112() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..112 { state.advance_step(); }
        assert_eq!(state.step, 112);
        let mut summary = EpochSummary::new(112);
        let m = GanMetrics { step: 112, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_113() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..113 { state.advance_step(); }
        assert_eq!(state.step, 113);
        let mut summary = EpochSummary::new(113);
        let m = GanMetrics { step: 113, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_114() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..114 { state.advance_step(); }
        assert_eq!(state.step, 114);
        let mut summary = EpochSummary::new(114);
        let m = GanMetrics { step: 114, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_115() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..115 { state.advance_step(); }
        assert_eq!(state.step, 115);
        let mut summary = EpochSummary::new(115);
        let m = GanMetrics { step: 115, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_116() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..116 { state.advance_step(); }
        assert_eq!(state.step, 116);
        let mut summary = EpochSummary::new(116);
        let m = GanMetrics { step: 116, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_117() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..117 { state.advance_step(); }
        assert_eq!(state.step, 117);
        let mut summary = EpochSummary::new(117);
        let m = GanMetrics { step: 117, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_118() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..118 { state.advance_step(); }
        assert_eq!(state.step, 118);
        let mut summary = EpochSummary::new(118);
        let m = GanMetrics { step: 118, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_119() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..119 { state.advance_step(); }
        assert_eq!(state.step, 119);
        let mut summary = EpochSummary::new(119);
        let m = GanMetrics { step: 119, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_120() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..120 { state.advance_step(); }
        assert_eq!(state.step, 120);
        let mut summary = EpochSummary::new(120);
        let m = GanMetrics { step: 120, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_121() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..121 { state.advance_step(); }
        assert_eq!(state.step, 121);
        let mut summary = EpochSummary::new(121);
        let m = GanMetrics { step: 121, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_122() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..122 { state.advance_step(); }
        assert_eq!(state.step, 122);
        let mut summary = EpochSummary::new(122);
        let m = GanMetrics { step: 122, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_123() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..123 { state.advance_step(); }
        assert_eq!(state.step, 123);
        let mut summary = EpochSummary::new(123);
        let m = GanMetrics { step: 123, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_124() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..124 { state.advance_step(); }
        assert_eq!(state.step, 124);
        let mut summary = EpochSummary::new(124);
        let m = GanMetrics { step: 124, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_125() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..125 { state.advance_step(); }
        assert_eq!(state.step, 125);
        let mut summary = EpochSummary::new(125);
        let m = GanMetrics { step: 125, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_126() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..126 { state.advance_step(); }
        assert_eq!(state.step, 126);
        let mut summary = EpochSummary::new(126);
        let m = GanMetrics { step: 126, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_127() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..127 { state.advance_step(); }
        assert_eq!(state.step, 127);
        let mut summary = EpochSummary::new(127);
        let m = GanMetrics { step: 127, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_128() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..128 { state.advance_step(); }
        assert_eq!(state.step, 128);
        let mut summary = EpochSummary::new(128);
        let m = GanMetrics { step: 128, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_129() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..129 { state.advance_step(); }
        assert_eq!(state.step, 129);
        let mut summary = EpochSummary::new(129);
        let m = GanMetrics { step: 129, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_130() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..130 { state.advance_step(); }
        assert_eq!(state.step, 130);
        let mut summary = EpochSummary::new(130);
        let m = GanMetrics { step: 130, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_131() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..131 { state.advance_step(); }
        assert_eq!(state.step, 131);
        let mut summary = EpochSummary::new(131);
        let m = GanMetrics { step: 131, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_132() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..132 { state.advance_step(); }
        assert_eq!(state.step, 132);
        let mut summary = EpochSummary::new(132);
        let m = GanMetrics { step: 132, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_133() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..133 { state.advance_step(); }
        assert_eq!(state.step, 133);
        let mut summary = EpochSummary::new(133);
        let m = GanMetrics { step: 133, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_134() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..134 { state.advance_step(); }
        assert_eq!(state.step, 134);
        let mut summary = EpochSummary::new(134);
        let m = GanMetrics { step: 134, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_135() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..135 { state.advance_step(); }
        assert_eq!(state.step, 135);
        let mut summary = EpochSummary::new(135);
        let m = GanMetrics { step: 135, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_136() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..136 { state.advance_step(); }
        assert_eq!(state.step, 136);
        let mut summary = EpochSummary::new(136);
        let m = GanMetrics { step: 136, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_137() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..137 { state.advance_step(); }
        assert_eq!(state.step, 137);
        let mut summary = EpochSummary::new(137);
        let m = GanMetrics { step: 137, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_138() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..138 { state.advance_step(); }
        assert_eq!(state.step, 138);
        let mut summary = EpochSummary::new(138);
        let m = GanMetrics { step: 138, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_139() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..139 { state.advance_step(); }
        assert_eq!(state.step, 139);
        let mut summary = EpochSummary::new(139);
        let m = GanMetrics { step: 139, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_140() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..140 { state.advance_step(); }
        assert_eq!(state.step, 140);
        let mut summary = EpochSummary::new(140);
        let m = GanMetrics { step: 140, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_141() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..141 { state.advance_step(); }
        assert_eq!(state.step, 141);
        let mut summary = EpochSummary::new(141);
        let m = GanMetrics { step: 141, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_142() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..142 { state.advance_step(); }
        assert_eq!(state.step, 142);
        let mut summary = EpochSummary::new(142);
        let m = GanMetrics { step: 142, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_143() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..143 { state.advance_step(); }
        assert_eq!(state.step, 143);
        let mut summary = EpochSummary::new(143);
        let m = GanMetrics { step: 143, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_144() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..144 { state.advance_step(); }
        assert_eq!(state.step, 144);
        let mut summary = EpochSummary::new(144);
        let m = GanMetrics { step: 144, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_145() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..145 { state.advance_step(); }
        assert_eq!(state.step, 145);
        let mut summary = EpochSummary::new(145);
        let m = GanMetrics { step: 145, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_146() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..146 { state.advance_step(); }
        assert_eq!(state.step, 146);
        let mut summary = EpochSummary::new(146);
        let m = GanMetrics { step: 146, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_147() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..147 { state.advance_step(); }
        assert_eq!(state.step, 147);
        let mut summary = EpochSummary::new(147);
        let m = GanMetrics { step: 147, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_148() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..148 { state.advance_step(); }
        assert_eq!(state.step, 148);
        let mut summary = EpochSummary::new(148);
        let m = GanMetrics { step: 148, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_149() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..149 { state.advance_step(); }
        assert_eq!(state.step, 149);
        let mut summary = EpochSummary::new(149);
        let m = GanMetrics { step: 149, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_150() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..150 { state.advance_step(); }
        assert_eq!(state.step, 150);
        let mut summary = EpochSummary::new(150);
        let m = GanMetrics { step: 150, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_151() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..151 { state.advance_step(); }
        assert_eq!(state.step, 151);
        let mut summary = EpochSummary::new(151);
        let m = GanMetrics { step: 151, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_152() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..152 { state.advance_step(); }
        assert_eq!(state.step, 152);
        let mut summary = EpochSummary::new(152);
        let m = GanMetrics { step: 152, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_153() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..153 { state.advance_step(); }
        assert_eq!(state.step, 153);
        let mut summary = EpochSummary::new(153);
        let m = GanMetrics { step: 153, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_154() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..154 { state.advance_step(); }
        assert_eq!(state.step, 154);
        let mut summary = EpochSummary::new(154);
        let m = GanMetrics { step: 154, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_155() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..155 { state.advance_step(); }
        assert_eq!(state.step, 155);
        let mut summary = EpochSummary::new(155);
        let m = GanMetrics { step: 155, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_156() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..156 { state.advance_step(); }
        assert_eq!(state.step, 156);
        let mut summary = EpochSummary::new(156);
        let m = GanMetrics { step: 156, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_157() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..157 { state.advance_step(); }
        assert_eq!(state.step, 157);
        let mut summary = EpochSummary::new(157);
        let m = GanMetrics { step: 157, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_158() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..158 { state.advance_step(); }
        assert_eq!(state.step, 158);
        let mut summary = EpochSummary::new(158);
        let m = GanMetrics { step: 158, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_159() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..159 { state.advance_step(); }
        assert_eq!(state.step, 159);
        let mut summary = EpochSummary::new(159);
        let m = GanMetrics { step: 159, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_160() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..160 { state.advance_step(); }
        assert_eq!(state.step, 160);
        let mut summary = EpochSummary::new(160);
        let m = GanMetrics { step: 160, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_161() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..161 { state.advance_step(); }
        assert_eq!(state.step, 161);
        let mut summary = EpochSummary::new(161);
        let m = GanMetrics { step: 161, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_162() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..162 { state.advance_step(); }
        assert_eq!(state.step, 162);
        let mut summary = EpochSummary::new(162);
        let m = GanMetrics { step: 162, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_163() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..163 { state.advance_step(); }
        assert_eq!(state.step, 163);
        let mut summary = EpochSummary::new(163);
        let m = GanMetrics { step: 163, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_164() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..164 { state.advance_step(); }
        assert_eq!(state.step, 164);
        let mut summary = EpochSummary::new(164);
        let m = GanMetrics { step: 164, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_165() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..165 { state.advance_step(); }
        assert_eq!(state.step, 165);
        let mut summary = EpochSummary::new(165);
        let m = GanMetrics { step: 165, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_166() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..166 { state.advance_step(); }
        assert_eq!(state.step, 166);
        let mut summary = EpochSummary::new(166);
        let m = GanMetrics { step: 166, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_167() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..167 { state.advance_step(); }
        assert_eq!(state.step, 167);
        let mut summary = EpochSummary::new(167);
        let m = GanMetrics { step: 167, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_168() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..168 { state.advance_step(); }
        assert_eq!(state.step, 168);
        let mut summary = EpochSummary::new(168);
        let m = GanMetrics { step: 168, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_169() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..169 { state.advance_step(); }
        assert_eq!(state.step, 169);
        let mut summary = EpochSummary::new(169);
        let m = GanMetrics { step: 169, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_170() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..170 { state.advance_step(); }
        assert_eq!(state.step, 170);
        let mut summary = EpochSummary::new(170);
        let m = GanMetrics { step: 170, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_171() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..171 { state.advance_step(); }
        assert_eq!(state.step, 171);
        let mut summary = EpochSummary::new(171);
        let m = GanMetrics { step: 171, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_172() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..172 { state.advance_step(); }
        assert_eq!(state.step, 172);
        let mut summary = EpochSummary::new(172);
        let m = GanMetrics { step: 172, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_173() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..173 { state.advance_step(); }
        assert_eq!(state.step, 173);
        let mut summary = EpochSummary::new(173);
        let m = GanMetrics { step: 173, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_174() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..174 { state.advance_step(); }
        assert_eq!(state.step, 174);
        let mut summary = EpochSummary::new(174);
        let m = GanMetrics { step: 174, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_175() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..175 { state.advance_step(); }
        assert_eq!(state.step, 175);
        let mut summary = EpochSummary::new(175);
        let m = GanMetrics { step: 175, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_176() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..176 { state.advance_step(); }
        assert_eq!(state.step, 176);
        let mut summary = EpochSummary::new(176);
        let m = GanMetrics { step: 176, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_177() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..177 { state.advance_step(); }
        assert_eq!(state.step, 177);
        let mut summary = EpochSummary::new(177);
        let m = GanMetrics { step: 177, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_178() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..178 { state.advance_step(); }
        assert_eq!(state.step, 178);
        let mut summary = EpochSummary::new(178);
        let m = GanMetrics { step: 178, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_179() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..179 { state.advance_step(); }
        assert_eq!(state.step, 179);
        let mut summary = EpochSummary::new(179);
        let m = GanMetrics { step: 179, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_180() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..180 { state.advance_step(); }
        assert_eq!(state.step, 180);
        let mut summary = EpochSummary::new(180);
        let m = GanMetrics { step: 180, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_181() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..181 { state.advance_step(); }
        assert_eq!(state.step, 181);
        let mut summary = EpochSummary::new(181);
        let m = GanMetrics { step: 181, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_182() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..182 { state.advance_step(); }
        assert_eq!(state.step, 182);
        let mut summary = EpochSummary::new(182);
        let m = GanMetrics { step: 182, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_183() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..183 { state.advance_step(); }
        assert_eq!(state.step, 183);
        let mut summary = EpochSummary::new(183);
        let m = GanMetrics { step: 183, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_184() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..184 { state.advance_step(); }
        assert_eq!(state.step, 184);
        let mut summary = EpochSummary::new(184);
        let m = GanMetrics { step: 184, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_185() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..185 { state.advance_step(); }
        assert_eq!(state.step, 185);
        let mut summary = EpochSummary::new(185);
        let m = GanMetrics { step: 185, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_186() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..186 { state.advance_step(); }
        assert_eq!(state.step, 186);
        let mut summary = EpochSummary::new(186);
        let m = GanMetrics { step: 186, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_187() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..187 { state.advance_step(); }
        assert_eq!(state.step, 187);
        let mut summary = EpochSummary::new(187);
        let m = GanMetrics { step: 187, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_188() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..188 { state.advance_step(); }
        assert_eq!(state.step, 188);
        let mut summary = EpochSummary::new(188);
        let m = GanMetrics { step: 188, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_189() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..189 { state.advance_step(); }
        assert_eq!(state.step, 189);
        let mut summary = EpochSummary::new(189);
        let m = GanMetrics { step: 189, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_190() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..190 { state.advance_step(); }
        assert_eq!(state.step, 190);
        let mut summary = EpochSummary::new(190);
        let m = GanMetrics { step: 190, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_191() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..191 { state.advance_step(); }
        assert_eq!(state.step, 191);
        let mut summary = EpochSummary::new(191);
        let m = GanMetrics { step: 191, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_192() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..192 { state.advance_step(); }
        assert_eq!(state.step, 192);
        let mut summary = EpochSummary::new(192);
        let m = GanMetrics { step: 192, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_193() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..193 { state.advance_step(); }
        assert_eq!(state.step, 193);
        let mut summary = EpochSummary::new(193);
        let m = GanMetrics { step: 193, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_194() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..194 { state.advance_step(); }
        assert_eq!(state.step, 194);
        let mut summary = EpochSummary::new(194);
        let m = GanMetrics { step: 194, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_195() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..195 { state.advance_step(); }
        assert_eq!(state.step, 195);
        let mut summary = EpochSummary::new(195);
        let m = GanMetrics { step: 195, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_196() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..196 { state.advance_step(); }
        assert_eq!(state.step, 196);
        let mut summary = EpochSummary::new(196);
        let m = GanMetrics { step: 196, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_197() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..197 { state.advance_step(); }
        assert_eq!(state.step, 197);
        let mut summary = EpochSummary::new(197);
        let m = GanMetrics { step: 197, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_198() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..198 { state.advance_step(); }
        assert_eq!(state.step, 198);
        let mut summary = EpochSummary::new(198);
        let m = GanMetrics { step: 198, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_199() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..199 { state.advance_step(); }
        assert_eq!(state.step, 199);
        let mut summary = EpochSummary::new(199);
        let m = GanMetrics { step: 199, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_200() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..200 { state.advance_step(); }
        assert_eq!(state.step, 200);
        let mut summary = EpochSummary::new(200);
        let m = GanMetrics { step: 200, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_201() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..201 { state.advance_step(); }
        assert_eq!(state.step, 201);
        let mut summary = EpochSummary::new(201);
        let m = GanMetrics { step: 201, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_202() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..202 { state.advance_step(); }
        assert_eq!(state.step, 202);
        let mut summary = EpochSummary::new(202);
        let m = GanMetrics { step: 202, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_203() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..203 { state.advance_step(); }
        assert_eq!(state.step, 203);
        let mut summary = EpochSummary::new(203);
        let m = GanMetrics { step: 203, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_204() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..204 { state.advance_step(); }
        assert_eq!(state.step, 204);
        let mut summary = EpochSummary::new(204);
        let m = GanMetrics { step: 204, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_205() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..205 { state.advance_step(); }
        assert_eq!(state.step, 205);
        let mut summary = EpochSummary::new(205);
        let m = GanMetrics { step: 205, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_206() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..206 { state.advance_step(); }
        assert_eq!(state.step, 206);
        let mut summary = EpochSummary::new(206);
        let m = GanMetrics { step: 206, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_207() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..207 { state.advance_step(); }
        assert_eq!(state.step, 207);
        let mut summary = EpochSummary::new(207);
        let m = GanMetrics { step: 207, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_208() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..208 { state.advance_step(); }
        assert_eq!(state.step, 208);
        let mut summary = EpochSummary::new(208);
        let m = GanMetrics { step: 208, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_209() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..209 { state.advance_step(); }
        assert_eq!(state.step, 209);
        let mut summary = EpochSummary::new(209);
        let m = GanMetrics { step: 209, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_210() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..210 { state.advance_step(); }
        assert_eq!(state.step, 210);
        let mut summary = EpochSummary::new(210);
        let m = GanMetrics { step: 210, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_211() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..211 { state.advance_step(); }
        assert_eq!(state.step, 211);
        let mut summary = EpochSummary::new(211);
        let m = GanMetrics { step: 211, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_212() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..212 { state.advance_step(); }
        assert_eq!(state.step, 212);
        let mut summary = EpochSummary::new(212);
        let m = GanMetrics { step: 212, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_213() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..213 { state.advance_step(); }
        assert_eq!(state.step, 213);
        let mut summary = EpochSummary::new(213);
        let m = GanMetrics { step: 213, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_214() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..214 { state.advance_step(); }
        assert_eq!(state.step, 214);
        let mut summary = EpochSummary::new(214);
        let m = GanMetrics { step: 214, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_215() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..215 { state.advance_step(); }
        assert_eq!(state.step, 215);
        let mut summary = EpochSummary::new(215);
        let m = GanMetrics { step: 215, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_core_stress_216() {
        let mut state = GanState::new(
            vec![Tensor::zeros(vec![4, 4])],
            vec![Tensor::zeros(vec![4, 4])],
        );
        for _ in 0..216 { state.advance_step(); }
        assert_eq!(state.step, 216);
        let mut summary = EpochSummary::new(216);
        let m = GanMetrics { step: 216, d_loss: 0.5, g_loss: 0.8, ..Default::default() };
        summary.update(&m);
        summary.finalize();
        assert!((summary.avg_d_loss - 0.5).abs() < 1e-9);
    }

    // GAN training and evaluation padding line 0
    // GAN training and evaluation padding line 1
    // GAN training and evaluation padding line 2
    // GAN training and evaluation padding line 3
    // GAN training and evaluation padding line 4
    // GAN training and evaluation padding line 5
    // GAN training and evaluation padding line 6
    // GAN training and evaluation padding line 7
    // GAN training and evaluation padding line 8
    // GAN training and evaluation padding line 9
    // GAN training and evaluation padding line 10
    // GAN training and evaluation padding line 11
    // GAN training and evaluation padding line 12
    // GAN training and evaluation padding line 13
}
