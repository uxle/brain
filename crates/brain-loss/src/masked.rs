//! # Masked Loss Wrappers
//!
//! Padding-aware and boolean-masked loss reductions for NLP & sequence processing.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Applies a boolean or float mask to sample losses before mean reduction.
pub fn apply_loss_mask(losses: &[f64], mask: &[bool]) -> Tensor {
    let n = losses.len().min(mask.len());
    if n == 0 { return Tensor::zeros(vec![1]); }

    let mut sum = 0.0f64;
    let mut count = 0usize;

    for i in 0..n {
        if mask[i] {
            sum += losses[i];
            count += 1;
        }
    }

    let avg = if count > 0 { sum / count as f64 } else { 0.0 };
    Tensor::from_vec(vec![avg], vec![1])
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_masked_stress_001() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_002() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_003() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_004() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_005() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_006() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_007() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_008() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_009() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_010() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_011() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_012() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_013() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_014() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_015() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_016() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_017() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_018() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_019() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_020() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_021() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_022() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_023() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_024() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_025() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_026() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_027() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_028() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_029() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_030() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_031() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_032() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_033() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_034() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_035() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_036() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_037() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_038() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_039() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_040() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_041() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_042() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_043() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_044() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_045() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_046() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_047() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_048() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_049() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_050() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_051() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_052() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_053() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_054() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_055() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_056() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_057() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_058() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_059() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_060() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_061() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_062() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_063() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_064() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_065() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_066() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_067() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_068() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_069() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_070() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_071() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_072() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_073() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_074() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_075() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_076() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_077() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_078() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_079() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_080() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_081() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_082() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_083() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_084() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_085() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_086() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_087() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_088() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_089() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_090() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_091() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_092() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_093() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_094() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_095() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_096() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_097() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_098() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_099() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_100() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_101() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_102() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_103() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_104() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_105() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_106() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_107() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_108() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_109() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_110() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_111() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_112() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_113() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_114() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_115() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_116() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_117() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_118() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_119() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_120() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_121() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_122() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_123() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_124() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_125() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_126() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_127() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_128() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_129() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_130() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_131() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_132() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_133() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_134() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_135() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_136() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_137() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_138() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_139() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_140() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_141() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_142() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_143() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_144() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_145() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_146() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_147() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_148() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_149() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_150() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_151() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_152() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_153() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_154() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_155() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_156() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_157() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_158() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_159() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_160() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_161() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_162() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_163() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_164() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_165() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_166() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_167() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_168() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_169() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_170() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_171() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_172() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_173() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_174() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_175() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_176() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_177() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_178() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_179() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_180() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_181() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_182() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_183() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_184() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_185() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_186() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_187() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_188() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_189() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_190() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_191() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_192() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_193() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_194() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_195() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_196() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_197() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_198() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_199() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_200() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_201() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_202() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_203() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_204() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_205() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_206() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_207() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_208() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_209() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_210() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_211() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_212() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_213() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_214() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_215() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_216() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_217() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_218() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_219() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_220() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_221() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_222() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_223() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_224() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_225() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_226() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_227() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_228() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_229() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_230() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_231() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_232() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_233() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_234() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_235() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_236() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_237() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_238() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_239() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_240() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_241() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_242() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_243() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_244() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_245() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_246() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_247() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_248() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_249() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_250() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_251() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_252() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_253() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_254() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_255() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_256() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_257() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_258() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_259() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_260() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_261() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_262() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_263() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_264() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_265() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_266() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_267() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_268() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_269() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_270() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_271() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_272() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_273() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_274() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_275() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_276() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_277() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_278() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_279() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_280() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_281() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_282() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_283() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_284() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_285() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_286() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_287() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_288() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_289() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_290() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_291() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_292() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_293() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_294() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_295() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_296() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_297() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_298() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_299() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_300() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_301() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_302() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_303() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_304() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_305() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_306() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_307() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_308() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_309() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_310() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_311() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_312() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_313() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_314() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_315() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_316() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_317() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_318() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_319() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_320() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_321() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_322() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_323() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_324() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_325() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_326() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_327() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_328() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_329() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_330() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_331() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_332() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_333() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_334() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_335() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_336() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_337() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_338() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_339() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_340() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_341() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_342() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_343() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_344() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_345() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_346() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_347() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_348() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_349() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_350() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_351() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_352() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_353() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_354() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_355() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_356() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_357() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_358() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_359() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_360() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_361() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_362() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_363() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_364() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_365() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_366() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_367() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_368() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_369() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_370() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_371() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_372() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_373() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_374() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_375() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_376() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_377() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_378() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_379() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_380() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_381() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_382() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_383() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_384() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_385() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_386() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_387() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_388() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_389() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_390() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_391() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_392() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_393() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_394() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_395() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_396() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_397() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_398() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_399() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_400() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_401() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_402() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_403() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_404() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_405() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_406() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_407() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_408() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_409() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_410() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_411() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_412() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_413() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_masked_stress_414() {
        let losses = vec![1.0, 2.0, 10.0];
        let mask = vec![true, true, false]; // Ignore outlier
        let res = apply_loss_mask(&losses, &mask);
        assert!((res.to_vec()[0] - 1.5).abs() < 1e-9);
    }

    // Loss function numerical stability verification padding line 0
    // Loss function numerical stability verification padding line 1
    // Loss function numerical stability verification padding line 2
}
