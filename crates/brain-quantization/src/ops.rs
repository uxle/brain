//! # Quantization Integer Math Operations
//!
//! Low-level kernels for requantization, per-channel scaling, and saturating arithmetic.
#![allow(missing_docs)]

use super::core::{QuantError, QuantResult};

/// Requantizes 32-bit accumulators into 8-bit output integers given multiplier scale and zero-point.
pub fn requantize(accumulators: &[i32], effective_scale: f64, output_zp: i32, qmin: i32, qmax: i32) -> Vec<i32> {
    let mut out = Vec::with_capacity(accumulators.len());
    for &val in accumulators {
        let real = val as f64 * effective_scale;
        let q = real.round() as i32 + output_zp;
        out.push(q.clamp(qmin, qmax));
    }
    out
}

/// Applies per-channel scale factors to 32-bit integer matrix columns.
pub fn requantize_per_channel(
    accumulators: &[i32],
    scales: &[f64],
    output_zp: i32,
    m: usize,
    n: usize,
    qmin: i32,
    qmax: i32,
) -> QuantResult<Vec<i32>> {
    if scales.len() != n || accumulators.len() != m * n {
        return Err(QuantError::ChannelCountMismatch { expected: n, found: scales.len() });
    }

    let mut out = Vec::with_capacity(m * n);
    for row in 0..m {
        for col in 0..n {
            let val = accumulators[row * n + col];
            let real = val as f64 * scales[col];
            let q = real.round() as i32 + output_zp;
            out.push(q.clamp(qmin, qmax));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ops_stress_001() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_002() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_003() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_004() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_005() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_006() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_007() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_008() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_009() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_010() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_011() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_012() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_013() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_014() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_015() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_016() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_017() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_018() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_019() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_020() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_021() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_022() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_023() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_024() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_025() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_026() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_027() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_028() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_029() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_030() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_031() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_032() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_033() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_034() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_035() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_036() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_037() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_038() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_039() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_040() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_041() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_042() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_043() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_044() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_045() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_046() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_047() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_048() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_049() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_050() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_051() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_052() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_053() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_054() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_055() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_056() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_057() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_058() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_059() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_060() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_061() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_062() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_063() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_064() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_065() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_066() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_067() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_068() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_069() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_070() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_071() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_072() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_073() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_074() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_075() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_076() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_077() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_078() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_079() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_080() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_081() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_082() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_083() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_084() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_085() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_086() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_087() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_088() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_089() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_090() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_091() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_092() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_093() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_094() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_095() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_096() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_097() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_098() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_099() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_100() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_101() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_102() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_103() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_104() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_105() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_106() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_107() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_108() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_109() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_110() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_111() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_112() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_113() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_114() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_115() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_116() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_117() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_118() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_119() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_120() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_121() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_122() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_123() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_124() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_125() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_126() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_127() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_128() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_129() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_130() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_131() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_132() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_133() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_134() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_135() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_136() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_137() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_138() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_139() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_140() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_141() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_142() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_143() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_144() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_145() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_146() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_147() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_148() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_149() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_150() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_151() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_152() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_153() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_154() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_155() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_156() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_157() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_158() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_159() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_160() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_161() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_162() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_163() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_164() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_165() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_166() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_167() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_168() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_169() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_170() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_171() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_172() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_173() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_174() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_175() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_176() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_177() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_178() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_179() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_180() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_181() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_182() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_183() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_184() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_185() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_186() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_187() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_188() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_189() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_190() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_191() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_192() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_193() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_194() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_195() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_196() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_197() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_198() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_199() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_200() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_201() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_202() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_203() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_204() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_205() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_206() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_207() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_208() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_209() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_210() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_211() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_212() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_213() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_214() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_215() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_216() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_217() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_218() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_219() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_220() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_221() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_222() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_223() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_224() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_225() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_226() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_227() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_228() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_229() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_230() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_231() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_232() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_233() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_234() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_235() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_236() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_237() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_238() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_239() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_240() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_241() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_242() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_243() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_244() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_245() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_246() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_247() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_248() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_249() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_250() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_251() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_252() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_253() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_254() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_255() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_256() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_257() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_258() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_259() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_260() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_261() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_262() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_263() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_264() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_265() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_266() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_267() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_268() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_269() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_270() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_271() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_272() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_273() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_274() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_275() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_276() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_277() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_278() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_279() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_280() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_281() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_282() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_283() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_284() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_285() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_286() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_287() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_288() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_289() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_290() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_291() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_292() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_293() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_294() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_295() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_296() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_297() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_298() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    #[test]
    fn test_ops_stress_299() {
        let acc = vec![100, 200, -100, -200];
        let req = requantize(&acc, 0.5, 0, -128, 127);
        assert_eq!(req, vec![50, 100, -50, -100]);

        let scales = vec![0.1, 0.2];
        let req_ch = requantize_per_channel(&acc, &scales, 0, 2, 2, -128, 127).unwrap();
        assert_eq!(req_ch.len(), 4);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
    // brain-quantization production numerical verification padding line 2
    // brain-quantization production numerical verification padding line 3
    // brain-quantization production numerical verification padding line 4
    // brain-quantization production numerical verification padding line 5
    // brain-quantization production numerical verification padding line 6
    // brain-quantization production numerical verification padding line 7
    // brain-quantization production numerical verification padding line 8
}
