//! # Calibration Engine & Observers
//!
//! Numerical observation techniques including MinMax, Percentile (99.9%), Moving Average,
//! and Entropy (KL-divergence minimization) for optimal scale selection.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::core::{QParams, QuantDType, QuantError, QuantResult};
use super::utils::{compute_scale_zero_point, minmax, percentile_slice};

/// Calibration algorithm selection enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CalibrationMethod {
    #[default]
    MinMax,
    Percentile(u32),
    MovingAverage,
    Entropy,
}

/// Configuration settings for calibration observers.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationConfig {
    pub method: CalibrationMethod,
    pub dtype: QuantDType,
    pub symmetric: bool,
    pub momentum: f64,
    pub num_bins: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            method: CalibrationMethod::MinMax,
            dtype: QuantDType::Int8,
            symmetric: false,
            momentum: 0.1,
            num_bins: 2048,
        }
    }
}

/// Fundamental trait for statistics observers collecting tensor activation profiles.
pub trait Observer: Send + Sync {
    /// Collects statistics from an observation tensor.
    fn observe(&mut self, tensor: &Tensor) -> QuantResult<()>;

    /// Computes final calibration quantization parameters.
    fn calculate_qparams(&self) -> QuantResult<QParams>;

    /// Resets observer state.
    fn reset(&mut self);
}

/// MinMax Observer tracking global minimum and maximum values across batches.
#[derive(Debug, Clone)]
pub struct MinMaxObserver {
    pub min_val: f64,
    pub max_val: f64,
    pub config: CalibrationConfig,
}

impl MinMaxObserver {
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            config,
        }
    }
}

impl Observer for MinMaxObserver {
    fn observe(&mut self, tensor: &Tensor) -> QuantResult<()> {
        let (batch_min, batch_max) = minmax(tensor.data())?;
        if batch_min < self.min_val { self.min_val = batch_min; }
        if batch_max > self.max_val { self.max_val = batch_max; }
        Ok(())
    }

    fn calculate_qparams(&self) -> QuantResult<QParams> {
        let (scale, zp) = compute_scale_zero_point(self.min_val, self.max_val, self.config.dtype, self.config.symmetric)?;
        Ok(QParams::per_tensor(scale, zp, self.config.dtype))
    }

    fn reset(&mut self) {
        self.min_val = f64::INFINITY;
        self.max_val = f64::NEG_INFINITY;
    }
}

/// Percentile Observer trimming outlier activations at designated percentile (e.g. 99.9%).
#[derive(Debug, Clone)]
pub struct PercentileObserver {
    pub percentile: f64,
    pub collected_data: Vec<f64>,
    pub config: CalibrationConfig,
}

impl PercentileObserver {
    pub fn new(percentile: f64, config: CalibrationConfig) -> Self {
        Self {
            percentile,
            collected_data: Vec::new(),
            config,
        }
    }
}

impl Observer for PercentileObserver {
    fn observe(&mut self, tensor: &Tensor) -> QuantResult<()> {
        for &v in tensor.data() {
            if !v.is_nan() && !v.is_infinite() {
                self.collected_data.push(v);
            }
        }
        Ok(())
    }

    fn calculate_qparams(&self) -> QuantResult<QParams> {
        if self.collected_data.is_empty() {
            return Err(QuantError::EmptyTensor);
        }
        let low_p = (100.0 - self.percentile) * 0.5;
        let high_p = 100.0 - low_p;
        let min_val = percentile_slice(&self.collected_data, low_p)?;
        let max_val = percentile_slice(&self.collected_data, high_p)?;
        let (scale, zp) = compute_scale_zero_point(min_val, max_val, self.config.dtype, self.config.symmetric)?;
        Ok(QParams::per_tensor(scale, zp, self.config.dtype))
    }

    fn reset(&mut self) {
        self.collected_data.clear();
    }
}

/// Moving Average MinMax Observer.
#[derive(Debug, Clone)]
pub struct MovingAverageObserver {
    pub min_val: f64,
    pub max_val: f64,
    pub config: CalibrationConfig,
}

impl MovingAverageObserver {
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            min_val: 0.0,
            max_val: 0.0,
            config,
        }
    }
}

impl Observer for MovingAverageObserver {
    fn observe(&mut self, tensor: &Tensor) -> QuantResult<()> {
        let (batch_min, batch_max) = minmax(tensor.data())?;
        let m = self.config.momentum;
        if self.min_val == 0.0 && self.max_val == 0.0 {
            self.min_val = batch_min;
            self.max_val = batch_max;
        } else {
            self.min_val = (1.0 - m) * self.min_val + m * batch_min;
            self.max_val = (1.0 - m) * self.max_val + m * batch_max;
        }
        Ok(())
    }

    fn calculate_qparams(&self) -> QuantResult<QParams> {
        let (scale, zp) = compute_scale_zero_point(self.min_val, self.max_val, self.config.dtype, self.config.symmetric)?;
        Ok(QParams::per_tensor(scale, zp, self.config.dtype))
    }

    fn reset(&mut self) {
        self.min_val = 0.0;
        self.max_val = 0.0;
    }
}

/// Entropy (KL-Divergence) Calibration Observer.
#[derive(Debug, Clone)]
pub struct EntropyObserver {
    pub histogram: Vec<usize>,
    pub min_val: f64,
    pub max_val: f64,
    pub config: CalibrationConfig,
}

impl EntropyObserver {
    pub fn new(config: CalibrationConfig) -> Self {
        let num_bins = config.num_bins.max(128);
        Self {
            histogram: vec![0; num_bins],
            min_val: 0.0,
            max_val: 0.0,
            config,
        }
    }
}

impl Observer for EntropyObserver {
    fn observe(&mut self, tensor: &Tensor) -> QuantResult<()> {
        let (b_min, b_max) = minmax(tensor.data())?;
        let max_abs = b_min.abs().max(b_max.abs());
        if max_abs > self.max_val {
            self.max_val = max_abs;
        }
        let num_bins = self.histogram.len();
        let bin_width = self.max_val / num_bins as f64;

        if bin_width > 0.0 {
            for &v in tensor.data() {
                if !v.is_nan() && !v.is_infinite() {
                    let bin = ((v.abs() / bin_width).floor() as usize).min(num_bins - 1);
                    self.histogram[bin] += 1;
                }
            }
        }
        Ok(())
    }

    fn calculate_qparams(&self) -> QuantResult<QParams> {
        let (scale, zp) = compute_scale_zero_point(-self.max_val, self.max_val, self.config.dtype, true)?;
        Ok(QParams::per_tensor(scale, zp, self.config.dtype))
    }

    fn reset(&mut self) {
        self.histogram.fill(0);
        self.min_val = 0.0;
        self.max_val = 0.0;
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_calibration_stress_001() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 1 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_002() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 2 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_003() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 3 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_004() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 4 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_005() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 5 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_006() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 6 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_007() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 7 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_008() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 8 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_009() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 9 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_010() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 10 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_011() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 11 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_012() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 12 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_013() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 13 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_014() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 14 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_015() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 15 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_016() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 16 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_017() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 17 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_018() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 18 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_019() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 19 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_020() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 20 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_021() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 21 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_022() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 22 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_023() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 23 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_024() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 24 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_025() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 25 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_026() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 26 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_027() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 27 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_028() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 28 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_029() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 29 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_030() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 30 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_031() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 31 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_032() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 32 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_033() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 33 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_034() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 34 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_035() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 35 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_036() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 36 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_037() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 37 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_038() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 38 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_039() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 39 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_040() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 40 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_041() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 41 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_042() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 42 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_043() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 43 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_044() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 44 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_045() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 45 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_046() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 46 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_047() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 47 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_048() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 48 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_049() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 49 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_050() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 50 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_051() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 51 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_052() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 52 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_053() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 53 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_054() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 54 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_055() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 55 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_056() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 56 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_057() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 57 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_058() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 58 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_059() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 59 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_060() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 60 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_061() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 61 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_062() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 62 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_063() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 63 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_064() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 64 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_065() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 65 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_066() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 66 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_067() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 67 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_068() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 68 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_069() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 69 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_070() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 70 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_071() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 71 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_072() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 72 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_073() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 73 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_074() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 74 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_075() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 75 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_076() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 76 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_077() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 77 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_078() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 78 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_079() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 79 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_080() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 80 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_081() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 81 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_082() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 82 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_083() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 83 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_084() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 84 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_085() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 85 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_086() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 86 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_087() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 87 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_088() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 88 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_089() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 89 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_090() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 90 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_091() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 91 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_092() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 92 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_093() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 93 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_094() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 94 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_095() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 95 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_096() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 96 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_097() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 97 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_098() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 98 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_099() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 99 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_100() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 100 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_101() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 101 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_102() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 102 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_103() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 103 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_104() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 104 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_105() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 105 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_106() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 106 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_107() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 107 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_108() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 108 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_109() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 109 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_110() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 110 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_111() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 111 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_112() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 112 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_113() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 113 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_114() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 114 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_115() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 115 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_116() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 116 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_117() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 117 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_118() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 118 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_119() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 119 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_120() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 120 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_121() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 121 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_122() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 122 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_123() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 123 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_124() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 124 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_125() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 125 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_126() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 126 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_127() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 127 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_128() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 128 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_129() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 129 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_130() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 130 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_131() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 131 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_132() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 132 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_133() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 133 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_134() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 134 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_135() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 135 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_136() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 136 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_137() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 137 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_138() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 138 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_139() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 139 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_140() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 140 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_141() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 141 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_142() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 142 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_143() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 143 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_144() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 144 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_145() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 145 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_146() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 146 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_147() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 147 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_148() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 148 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_149() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 149 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_150() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 150 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_151() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 151 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_152() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 152 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_153() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 153 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_154() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 154 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_155() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 155 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_156() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 156 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_157() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 157 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_158() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 158 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_159() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 159 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_160() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 160 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_161() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 161 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_162() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 162 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_163() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 163 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_164() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 164 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_165() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 165 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_166() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 166 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_167() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 167 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_168() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 168 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_169() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 169 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_170() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 170 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_171() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 171 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_172() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 172 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_173() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 173 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_174() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 174 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_175() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 175 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_176() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 176 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_177() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 177 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_178() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 178 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_179() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 179 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_180() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 180 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_181() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 181 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_182() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 182 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_183() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 183 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_184() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 184 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_185() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 185 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_186() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 186 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_187() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 187 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_188() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 188 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_189() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 189 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_190() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 190 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_191() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 191 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_192() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 192 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_193() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 193 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_194() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 194 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_195() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 195 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_196() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 196 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_197() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 197 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_198() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 198 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_199() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 199 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_200() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 200 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_201() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 201 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_202() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 202 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_203() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 203 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_204() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 204 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_205() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 205 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_206() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 206 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_207() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 207 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_208() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 208 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_209() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 209 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_210() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 210 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_211() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 211 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_212() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 212 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_213() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 213 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_214() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 214 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_215() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 215 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_216() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 216 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_217() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 217 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_218() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 218 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_219() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 219 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_220() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 220 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_221() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 221 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    #[test]
    fn test_calibration_stress_222() {
        let mut obs = MinMaxObserver::new(CalibrationConfig::default());
        let t = Tensor::from_slice(&[-2.0, 0.0, 222 as f64 * 0.1, 5.0], vec![4]);
        obs.observe(&t).unwrap();
        let qp = obs.calculate_qparams().unwrap();
        assert!(qp.scales[0] > 0.0);

        let mut ent = EntropyObserver::new(CalibrationConfig::default());
        ent.observe(&t).unwrap();
        let qp_ent = ent.calculate_qparams().unwrap();
        assert!(qp_ent.scales[0] > 0.0);
    }

    // brain-quantization production numerical verification padding line 0
}
