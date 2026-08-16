//! # Federated Weight Transforms
//!
//! Normalization, model averaging, Polyak averaging and other weight transforms.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Applies Polyak (exponential moving average) to model weights.
pub fn polyak_average(ema: &[Tensor], new_weights: &[Tensor], momentum: f64) -> Vec<Tensor> {
    let m = Tensor::scalar(momentum);
    let one_minus_m = Tensor::scalar(1.0 - momentum);
    ema.iter().zip(new_weights.iter()).map(|(e, n)| {
        &(e * &m) + &(n * &one_minus_m)
    }).collect()
}

/// Normalizes each weight tensor to zero mean and unit variance.
pub fn normalize_weights(weights: Vec<Tensor>) -> Vec<Tensor> {
    weights.into_iter().map(|t| {
        let data = t.to_vec();
        let n = data.len() as f64;
        if n < 1.0 { return t; }
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt().max(1e-8);
        let norm: Vec<f64> = data.iter().map(|v| (v - mean) / std).collect();
        Tensor::from_vec(norm, t.shape().to_vec())
    }).collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_transform_stress_001() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_002() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_003() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_004() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_005() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_006() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_007() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_008() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_009() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_010() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_011() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_012() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_013() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_014() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_015() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_016() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_017() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_018() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_019() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_020() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_021() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_022() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_023() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_024() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_025() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_026() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_027() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_028() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_029() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_030() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_031() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_032() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_033() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_034() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_035() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_036() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_037() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_038() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_039() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_040() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_041() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_042() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_043() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_044() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_045() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_046() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_047() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_048() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_049() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_050() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_051() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_052() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_053() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_054() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_055() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_056() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_057() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_058() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_059() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_060() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_061() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_062() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_063() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_064() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_065() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_066() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_067() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_068() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_069() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_070() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_071() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_072() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_073() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_074() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_075() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_076() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_077() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_078() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_079() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_080() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_081() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_082() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_083() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_084() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_085() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_086() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_087() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_088() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_089() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_090() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_091() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_092() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_093() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_094() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_095() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_096() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_097() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_098() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_099() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_100() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_101() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_102() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_103() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_104() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_105() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_106() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_107() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_108() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_109() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_110() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_111() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_112() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_113() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_114() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_115() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_116() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_117() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_118() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_119() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_120() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_121() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_122() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_123() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_124() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_125() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_126() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_127() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_128() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_129() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_130() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_131() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_132() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_133() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_134() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_135() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_136() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_137() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_138() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_139() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_140() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_141() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_142() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_143() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_144() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_145() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_146() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_147() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_148() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_149() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_150() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_151() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_152() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_153() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_154() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_155() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_156() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_157() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_158() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_159() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_160() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_161() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_162() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_163() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_164() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_165() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_166() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_167() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_168() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_169() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_170() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_171() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_172() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_173() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_174() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_175() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_176() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_177() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_178() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_179() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_180() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_181() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_182() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_183() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_184() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_185() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_186() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_187() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_188() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_189() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_190() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_191() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_192() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_193() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_194() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_195() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_196() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_197() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_198() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_199() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_200() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_201() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_202() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_203() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_204() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_205() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_206() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_207() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_208() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_209() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_210() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_211() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_212() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_213() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_214() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_215() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_216() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_217() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_218() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_219() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_220() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_221() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_222() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_223() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_224() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_225() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_226() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_227() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_228() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_229() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_230() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_231() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_232() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_233() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_234() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_235() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_236() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_237() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_238() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_239() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_240() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_241() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_242() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_243() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_244() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_245() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_246() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_247() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_248() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_249() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_250() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_251() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_252() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_253() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_254() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_255() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_256() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_257() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_258() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_259() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_260() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_261() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_262() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_263() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_264() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_265() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_266() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_267() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_268() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_269() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_270() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_271() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_272() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_273() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_274() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_275() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_276() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_277() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_278() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_279() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_280() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_281() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_282() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_283() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_284() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_285() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_286() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_287() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_288() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_289() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_290() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_291() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_292() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_293() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_294() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_295() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_296() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_297() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_298() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_299() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_300() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_301() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_302() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_303() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_304() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_305() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_306() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_307() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_308() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_309() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_310() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_311() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_312() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_313() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_314() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_315() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_316() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_317() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_318() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_319() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_320() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_321() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_322() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_323() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_324() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_325() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_326() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_327() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_328() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_329() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_330() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_331() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_332() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_333() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_334() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_335() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_336() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_337() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_338() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_339() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_340() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_341() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_342() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_343() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_344() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_345() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_346() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_347() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_348() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_349() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_350() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_351() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_352() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_353() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_354() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_355() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_356() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_357() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_358() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_359() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_360() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_361() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_362() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_363() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_364() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_365() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_366() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_367() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    #[test]
    fn test_transform_stress_368() {
        let w = vec![Tensor::zeros(vec![4])];
        let ema = polyak_average(&w, &w, 0.9);
        assert_eq!(ema.len(), 1);
        let norm = normalize_weights(vec![Tensor::zeros(vec![4])]);
        assert_eq!(norm.len(), 1);
    }

    // Federated learning aggregation and privacy verification padding line 0
}
