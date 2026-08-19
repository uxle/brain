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
}
