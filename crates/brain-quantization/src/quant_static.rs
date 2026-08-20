//! # Static Quantization Pipeline
//!
//! Offline calibration -> scale freezing -> integer-only execution.
#![allow(missing_docs)]

use super::calibration::{CalibrationConfig, MinMaxObserver, Observer};
use super::config::StaticConfig;
use super::core::{QParams, QuantError, QuantResult, QuantTensor};
use super::utils::quantize_val;
use brain_core::Tensor;

/// Static Quantization Manager coordinating calibration and layer quantization.
#[derive(Debug, Clone)]
pub struct StaticQuantizer {
    pub config: StaticConfig,
    pub observer: MinMaxObserver,
    pub calibrated_params: Option<QParams>,
}

impl StaticQuantizer {
    pub fn new(config: StaticConfig) -> Self {
        let cal_cfg = CalibrationConfig {
            dtype: config.activation_dtype,
            symmetric: false,
            ..Default::default()
        };
        Self {
            config,
            observer: MinMaxObserver::new(cal_cfg),
            calibrated_params: None,
        }
    }

    /// Feeds a calibration batch to update activation scale estimates.
    pub fn calibrate_batch(&mut self, batch: &Tensor) -> QuantResult<()> {
        self.observer.observe(batch)
    }

    /// Freezes observer statistics into static quantization parameters.
    pub fn freeze_calibration(&mut self) -> QuantResult<QParams> {
        let qparams = self.observer.calculate_qparams()?;
        self.calibrated_params = Some(qparams.clone());
        Ok(qparams)
    }

    /// Quantizes input using frozen static parameters.
    pub fn quantize_static(&self, tensor: &Tensor) -> QuantResult<QuantTensor> {
        let params = self.calibrated_params.as_ref().ok_or_else(|| {
            QuantError::CalibrationError("Static quantizer has not been calibrated".into())
        })?;

        let scale = params.scales[0];
        let zp = params.zero_points[0];
        let qmin = params.qmin;
        let qmax = params.qmax;

        let mut q_data = Vec::with_capacity(tensor.numel());
        for &v in tensor.data() {
            q_data.push(quantize_val(v, scale, zp, qmin, qmax));
        }

        Ok(QuantTensor::new(
            q_data,
            tensor.shape().to_vec(),
            params.clone(),
        ))
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
