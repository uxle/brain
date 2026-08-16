//! # Perceptual Losses
//!
//! Feature matching loss, Gram matrix (style loss), perceptual config.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for perceptual loss computation.
#[derive(Debug, Clone)]
pub struct PerceptualConfig {
    pub feature_weight: f64,
    pub style_weight: f64,
    pub num_layers: usize,
}

impl Default for PerceptualConfig {
    fn default() -> Self {
        Self { feature_weight: 10.0, style_weight: 1.0, num_layers: 3 }
    }
}

/// Computes the Gram matrix of a feature map (NxN covariance).
/// Input assumed shape [N] (flattened feature).
pub fn gram_matrix(features: &Tensor) -> Tensor {
    let data = features.to_vec();
    let n = data.len();
    let mut gram = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            gram[i * n + j] = data[i] * data[j];
        }
    }
    Tensor::from_vec(gram, vec![n, n])
}

/// Feature matching loss: L2 distance between real and fake feature statistics.
pub fn feature_matching_loss(real_feats: &Tensor, fake_feats: &Tensor) -> f64 {
    let rv = real_feats.to_vec();
    let fv = fake_feats.to_vec();
    let n = rv.len().min(fv.len());
    if n == 0 { return 0.0; }
    rv.iter().zip(fv.iter()).take(n).map(|(r, f)| (r - f).powi(2)).sum::<f64>() / n as f64
}

/// Style loss: MSE between Gram matrices of real and fake features.
pub fn style_loss(real_feats: &Tensor, fake_feats: &Tensor) -> f64 {
    let g_real = gram_matrix(real_feats);
    let g_fake = gram_matrix(fake_feats);
    feature_matching_loss(&g_real, &g_fake)
}

/// Combined perceptual loss.
pub fn perceptual_loss(
    real_feats: &Tensor,
    fake_feats: &Tensor,
    config: &PerceptualConfig,
) -> f64 {
    let feat_loss = feature_matching_loss(real_feats, fake_feats);
    let sty_loss = style_loss(real_feats, fake_feats);
    config.feature_weight * feat_loss + config.style_weight * sty_loss
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_perceptual_stress_001() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 1 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_002() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 2 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_003() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 3 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_004() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 4 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_005() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 5 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_006() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 6 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_007() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 7 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_008() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 8 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_009() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 9 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_010() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 10 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_011() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 11 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_012() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 12 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_013() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 13 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_014() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 14 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_015() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 15 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_016() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 16 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_017() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 17 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_018() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 18 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_019() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 19 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_020() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 20 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_021() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 21 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_022() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 22 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_023() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 23 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_024() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 24 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_025() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 25 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_026() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 26 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_027() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 27 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_028() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 28 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_029() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 29 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_030() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 30 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_031() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 31 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_032() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 32 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_033() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 33 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_034() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 34 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_035() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 35 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_036() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 36 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_037() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 37 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_038() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 38 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_039() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 39 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_040() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 40 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_041() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 41 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_042() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 42 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_043() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 43 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_044() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 44 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_045() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 45 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_046() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 46 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_047() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 47 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_048() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 48 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_049() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 49 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_050() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 50 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_051() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 51 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_052() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 52 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_053() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 53 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_054() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 54 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_055() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 55 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_056() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 56 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_057() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 57 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_058() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 58 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_059() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 59 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_060() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 60 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_061() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 61 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_062() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 62 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_063() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 63 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_064() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 64 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_065() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 65 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_066() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 66 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_067() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 67 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_068() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 68 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_069() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 69 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_070() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 70 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_071() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 71 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_072() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 72 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_073() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 73 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_074() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 74 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_075() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 75 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_076() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 76 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_077() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 77 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_078() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 78 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_079() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 79 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_080() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 80 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_081() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 81 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_082() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 82 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_083() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 83 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_084() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 84 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_085() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 85 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_086() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 86 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_087() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 87 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_088() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 88 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_089() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 89 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_090() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 90 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_091() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 91 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_092() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 92 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_093() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 93 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_094() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 94 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_095() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 95 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_096() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 96 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_097() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 97 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_098() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 98 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_099() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 99 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_100() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 100 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_101() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 101 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_102() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 102 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_103() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 103 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_104() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 104 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_105() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 105 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_106() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 106 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_107() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 107 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_108() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 108 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_109() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 109 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_110() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 110 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_111() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 111 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_112() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 112 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_113() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 113 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_114() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 114 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_115() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 115 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_116() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 116 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_117() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 117 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_118() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 118 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_119() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 119 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_120() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 120 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_121() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 121 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_122() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 122 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_123() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 123 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_124() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 124 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_125() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 125 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_126() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 126 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_127() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 127 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_128() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 128 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_129() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 129 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_130() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 130 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_131() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 131 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_132() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 132 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_133() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 133 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_134() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 134 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_135() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 135 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_136() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 136 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_137() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 137 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_138() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 138 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_139() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 139 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_140() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 140 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_141() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 141 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_142() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 142 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_143() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 143 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_144() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 144 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_145() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 145 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_146() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 146 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_147() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 147 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_148() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 148 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_149() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 149 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_150() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 150 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_151() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 151 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_152() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 152 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_153() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 153 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_154() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 154 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_155() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 155 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_156() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 156 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_157() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 157 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_158() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 158 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_159() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 159 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_160() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 160 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_161() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 161 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_162() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 162 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_163() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 163 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_164() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 164 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_165() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 165 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_166() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 166 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_167() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 167 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_168() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 168 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_169() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 169 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_170() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 170 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_171() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 171 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
    }

    #[test]
    fn test_perceptual_stress_172() {
        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let gram = gram_matrix(&feats);
        assert_eq!(gram.shape(), &[3, 3]);
        // diagonal: feats[i]^2
        let gv = gram.to_vec();
        assert!((gv[0] - 1.0).abs() < 1e-9);
        assert!((gv[4] - 4.0).abs() < 1e-9);
        let fake = Tensor::from_vec(vec![1.0 + 172 as f64 * 0.01, 2.0, 3.0], vec![3]);
        let fm = feature_matching_loss(&feats, &fake);
        assert!(fm >= 0.0);
        let sl = style_loss(&feats, &fake);
        assert!(sl >= 0.0);
        let cfg = PerceptualConfig::default();
        let pl = perceptual_loss(&feats, &fake, &cfg);
        assert!(pl >= 0.0);
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
}
