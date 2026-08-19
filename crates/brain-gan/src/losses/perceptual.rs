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
}
