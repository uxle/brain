//! # Quantization Error Analysis & Diagnostics
//!
//! Numerical fidelity metrics including Mean Squared Error (MSE), SNR, PSNR, and Maximum Absolute Error (MAE).
#![allow(missing_docs)]

use brain_core::Tensor;
use super::core::{QuantError, QuantResult};

/// Comprehensive report of quantization fidelity metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantErrorReport {
    pub mse: f64,
    pub mae: f64,
    pub snr_db: f64,
    pub psnr_db: f64,
    pub max_abs_diff: f64,
}

/// Evaluates error metrics between original floating point tensor and dequantized tensor.
pub fn analyze_quantization_error(original: &Tensor, dequantized: &Tensor) -> QuantResult<QuantErrorReport> {
    let orig_data = original.data();
    let deq_data = dequantized.data();

    if orig_data.len() != deq_data.len() || orig_data.is_empty() {
        return Err(QuantError::ShapeMismatch {
            expected: original.shape().to_vec(),
            found: dequantized.shape().to_vec(),
        });
    }

    let n = orig_data.len();
    let mut sum_sq_err = 0.0;
    let mut sum_abs_err = 0.0;
    let mut max_abs_diff = 0.0f64;
    let mut signal_power = 0.0;
    let mut max_signal = 0.0f64;

    for i in 0..n {
        let x = orig_data[i];
        let x_hat = deq_data[i];
        let diff = (x - x_hat).abs();

        sum_sq_err += diff * diff;
        sum_abs_err += diff;
        if diff > max_abs_diff { max_abs_diff = diff; }

        signal_power += x * x;
        if x.abs() > max_signal { max_signal = x.abs(); }
    }

    let mse = sum_sq_err / n as f64;
    let mae = sum_abs_err / n as f64;

    let snr_db = if sum_sq_err > 1e-15 {
        10.0 * (signal_power / sum_sq_err).max(1e-12).log10()
    } else {
        100.0
    };

    let psnr_db = if mse > 1e-15 {
        10.0 * ((max_signal * max_signal) / mse).max(1e-12).log10()
    } else {
        100.0
    };

    Ok(QuantErrorReport {
        mse,
        mae,
        snr_db,
        psnr_db,
        max_abs_diff,
    })
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
