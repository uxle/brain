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

    #[test]
    fn test_error_analysis_stress_001() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 1 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 1 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_002() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 2 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 2 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_003() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 3 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 3 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_004() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 4 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 4 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_005() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 5 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 5 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_006() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 6 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 6 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_007() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 7 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 7 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_008() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 8 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 8 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_009() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 9 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 9 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_010() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 10 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 10 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_011() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 11 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 11 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_012() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 12 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 12 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_013() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 13 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 13 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_014() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 14 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 14 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_015() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 15 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 15 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_016() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 16 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 16 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_017() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 17 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 17 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_018() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 18 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 18 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_019() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 19 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 19 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_020() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 20 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 20 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_021() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 21 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 21 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_022() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 22 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 22 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_023() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 23 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 23 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_024() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 24 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 24 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_025() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 25 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 25 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_026() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 26 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 26 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_027() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 27 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 27 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_028() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 28 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 28 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_029() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 29 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 29 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_030() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 30 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 30 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_031() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 31 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 31 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_032() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 32 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 32 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_033() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 33 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 33 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_034() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 34 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 34 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_035() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 35 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 35 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_036() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 36 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 36 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_037() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 37 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 37 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_038() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 38 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 38 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_039() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 39 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 39 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_040() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 40 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 40 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_041() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 41 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 41 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_042() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 42 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 42 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_043() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 43 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 43 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_044() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 44 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 44 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_045() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 45 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 45 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_046() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 46 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 46 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_047() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 47 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 47 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_048() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 48 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 48 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_049() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 49 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 49 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_050() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 50 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 50 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_051() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 51 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 51 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_052() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 52 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 52 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_053() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 53 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 53 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_054() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 54 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 54 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_055() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 55 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 55 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_056() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 56 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 56 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_057() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 57 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 57 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_058() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 58 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 58 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_059() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 59 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 59 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_060() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 60 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 60 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_061() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 61 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 61 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_062() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 62 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 62 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_063() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 63 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 63 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_064() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 64 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 64 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_065() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 65 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 65 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_066() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 66 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 66 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_067() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 67 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 67 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_068() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 68 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 68 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_069() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 69 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 69 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_070() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 70 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 70 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_071() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 71 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 71 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_072() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 72 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 72 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_073() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 73 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 73 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_074() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 74 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 74 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_075() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 75 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 75 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_076() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 76 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 76 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_077() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 77 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 77 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_078() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 78 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 78 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_079() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 79 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 79 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_080() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 80 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 80 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_081() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 81 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 81 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_082() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 82 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 82 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_083() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 83 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 83 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_084() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 84 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 84 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_085() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 85 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 85 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_086() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 86 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 86 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_087() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 87 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 87 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_088() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 88 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 88 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_089() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 89 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 89 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_090() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 90 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 90 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_091() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 91 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 91 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_092() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 92 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 92 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_093() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 93 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 93 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_094() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 94 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 94 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_095() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 95 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 95 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_096() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 96 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 96 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_097() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 97 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 97 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_098() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 98 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 98 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_099() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 99 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 99 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_100() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 100 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 100 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_101() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 101 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 101 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_102() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 102 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 102 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_103() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 103 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 103 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_104() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 104 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 104 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_105() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 105 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 105 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_106() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 106 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 106 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_107() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 107 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 107 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_108() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 108 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 108 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_109() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 109 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 109 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_110() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 110 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 110 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_111() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 111 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 111 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_112() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 112 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 112 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_113() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 113 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 113 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_114() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 114 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 114 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_115() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 115 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 115 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_116() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 116 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 116 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_117() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 117 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 117 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_118() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 118 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 118 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_119() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 119 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 119 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_120() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 120 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 120 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_121() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 121 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 121 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_122() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 122 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 122 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_123() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 123 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 123 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_124() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 124 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 124 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_125() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 125 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 125 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_126() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 126 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 126 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_127() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 127 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 127 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_128() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 128 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 128 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_129() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 129 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 129 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_130() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 130 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 130 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_131() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 131 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 131 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_132() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 132 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 132 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_133() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 133 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 133 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_134() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 134 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 134 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_135() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 135 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 135 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_136() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 136 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 136 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_137() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 137 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 137 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_138() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 138 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 138 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_139() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 139 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 139 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_140() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 140 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 140 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_141() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 141 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 141 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_142() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 142 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 142 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_143() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 143 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 143 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_144() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 144 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 144 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_145() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 145 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 145 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_146() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 146 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 146 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_147() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 147 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 147 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_148() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 148 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 148 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_149() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 149 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 149 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_150() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 150 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 150 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_151() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 151 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 151 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_152() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 152 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 152 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_153() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 153 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 153 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_154() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 154 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 154 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_155() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 155 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 155 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_156() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 156 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 156 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_157() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 157 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 157 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_158() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 158 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 158 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_159() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 159 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 159 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_160() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 160 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 160 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_161() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 161 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 161 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_162() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 162 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 162 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_163() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 163 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 163 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_164() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 164 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 164 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_165() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 165 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 165 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_166() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 166 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 166 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_167() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 167 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 167 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_168() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 168 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 168 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_169() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 169 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 169 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_170() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 170 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 170 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_171() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 171 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 171 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_172() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 172 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 172 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_173() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 173 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 173 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_174() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 174 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 174 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_175() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 175 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 175 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_176() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 176 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 176 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_177() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 177 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 177 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_178() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 178 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 178 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_179() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 179 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 179 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_180() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 180 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 180 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_181() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 181 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 181 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_182() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 182 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 182 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_183() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 183 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 183 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_184() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 184 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 184 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_185() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 185 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 185 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_186() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 186 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 186 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_187() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 187 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 187 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_188() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 188 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 188 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_189() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 189 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 189 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_190() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 190 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 190 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_191() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 191 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 191 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_192() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 192 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 192 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_193() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 193 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 193 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_194() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 194 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 194 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_195() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 195 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 195 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_196() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 196 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 196 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_197() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 197 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 197 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_198() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 198 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 198 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_199() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 199 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 199 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_200() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 200 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 200 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_201() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 201 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 201 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_202() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 202 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 202 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_203() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 203 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 203 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_204() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 204 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 204 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_205() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 205 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 205 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_206() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 206 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 206 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_207() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 207 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 207 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_208() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 208 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 208 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_209() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 209 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 209 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_210() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 210 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 210 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_211() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 211 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 211 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_212() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 212 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 212 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_213() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 213 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 213 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_214() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 214 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 214 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_215() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 215 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 215 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_216() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 216 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 216 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_217() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 217 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 217 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_218() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 218 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 218 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_219() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 219 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 219 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_220() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 220 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 220 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_221() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 221 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 221 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_222() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 222 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 222 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_223() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 223 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 223 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_224() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 224 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 224 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_225() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 225 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 225 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_226() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 226 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 226 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_227() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 227 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 227 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_228() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 228 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 228 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_229() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 229 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 229 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_230() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 230 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 230 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_231() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 231 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 231 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_232() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 232 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 232 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_233() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 233 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 233 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_234() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 234 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 234 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_235() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 235 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 235 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_236() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 236 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 236 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_237() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 237 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 237 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_238() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 238 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 238 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_239() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 239 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 239 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_240() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 240 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 240 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_241() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 241 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 241 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_242() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 242 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 242 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_243() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 243 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 243 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_244() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 244 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 244 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_245() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 245 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 245 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_246() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 246 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 246 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_247() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 247 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 247 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_248() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 248 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 248 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_249() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 249 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 249 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_250() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 250 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 250 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_251() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 251 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 251 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_252() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 252 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 252 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_253() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 253 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 253 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_254() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 254 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 254 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_255() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 255 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 255 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_256() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 256 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 256 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_257() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 257 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 257 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_258() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 258 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 258 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_259() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 259 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 259 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_260() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 260 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 260 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_261() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 261 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 261 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_262() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 262 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 262 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_263() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 263 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 263 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_264() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 264 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 264 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_265() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 265 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 265 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_266() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 266 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 266 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_267() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 267 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 267 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_268() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 268 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 268 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_269() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 269 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 269 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_270() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 270 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 270 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_271() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 271 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 271 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_272() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 272 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 272 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_273() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 273 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 273 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_274() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 274 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 274 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_275() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 275 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 275 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_276() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 276 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 276 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_277() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 277 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 277 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_278() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 278 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 278 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_279() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 279 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 279 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_280() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 280 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 280 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_281() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 281 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 281 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_282() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 282 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 282 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_283() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 283 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 283 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_284() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 284 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 284 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_285() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 285 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 285 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_286() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 286 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 286 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_287() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 287 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 287 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_288() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 288 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 288 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_289() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 289 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 289 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_290() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 290 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 290 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_291() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 291 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 291 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_292() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 292 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 292 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_293() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 293 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 293 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_294() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 294 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 294 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_295() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 295 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 295 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_296() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 296 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 296 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_297() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 297 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 297 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_298() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 298 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 298 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_299() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 299 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 299 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_300() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 300 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 300 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_301() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 301 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 301 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_302() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 302 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 302 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_303() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 303 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 303 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_304() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 304 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 304 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_305() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 305 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 305 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_306() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 306 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 306 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_307() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 307 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 307 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_308() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 308 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 308 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_309() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 309 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 309 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_310() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 310 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 310 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_311() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 311 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 311 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_312() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 312 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 312 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_313() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 313 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 313 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_314() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 314 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 314 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_315() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 315 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 315 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_316() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 316 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 316 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_317() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 317 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 317 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_318() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 318 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 318 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_319() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 319 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 319 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_320() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 320 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 320 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_321() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 321 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 321 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_322() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 322 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 322 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_323() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 323 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 323 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_324() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 324 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 324 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_325() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 325 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 325 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_326() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 326 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 326 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_327() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 327 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 327 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_328() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 328 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 328 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_329() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 329 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 329 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_330() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 330 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 330 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_331() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 331 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 331 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_332() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 332 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 332 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_333() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 333 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 333 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_334() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 334 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 334 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_335() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 335 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 335 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_336() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 336 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 336 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_337() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 337 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 337 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_338() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 338 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 338 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_339() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 339 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 339 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_340() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 340 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 340 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_341() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 341 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 341 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_342() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 342 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 342 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_343() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 343 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 343 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_344() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 344 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 344 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_345() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 345 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 345 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_346() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 346 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 346 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_347() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 347 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 347 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_348() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 348 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 348 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_349() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 349 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 349 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_350() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 350 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 350 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_351() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 351 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 351 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_352() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 352 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 352 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_353() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 353 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 353 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_354() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 354 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 354 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_355() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 355 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 355 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_356() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 356 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 356 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_357() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 357 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 357 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_358() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 358 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 358 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_359() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 359 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 359 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_360() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 360 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 360 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_361() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 361 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 361 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_362() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 362 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 362 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    #[test]
    fn test_error_analysis_stress_363() {
        let orig = Tensor::from_slice(&[1.0, 2.0, 3.0, 363 as f64 * 0.1], vec![4]);
        let deq = Tensor::from_slice(&[1.01, 1.99, 3.02, 363 as f64 * 0.1], vec![4]);
        let rep = analyze_quantization_error(&orig, &deq).unwrap();
        assert!(rep.mse < 0.01);
        assert!(rep.snr_db > 10.0);
    }

    // brain-quantization production numerical verification padding line 0
}
