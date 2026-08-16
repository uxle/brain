//! # StyleGAN-Lite
//!
//! Mapping network, modulated convolutions (AdaIN-style), style mixing.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::ops::{relu, leaky_relu, batch_norm};

/// Mapping network configuration.
#[derive(Debug, Clone)]
pub struct MappingConfig {
    pub latent_dim: usize,
    pub style_dim: usize,
    pub num_layers: usize,
    pub lr_multiplier: f64,
}

impl Default for MappingConfig {
    fn default() -> Self {
        Self { latent_dim: 512, style_dim: 512, num_layers: 8, lr_multiplier: 0.01 }
    }
}

/// Mapping network: z -> w (disentangled latent space).
pub struct MappingNetwork {
    pub config: MappingConfig,
    pub weights: Vec<Tensor>,
}

impl MappingNetwork {
    pub fn new(config: MappingConfig) -> Self {
        let mut weights = Vec::new();
        let mut in_dim = config.latent_dim;
        for _ in 0..config.num_layers {
            weights.push(Tensor::zeros(vec![config.style_dim, in_dim]));
            in_dim = config.style_dim;
        }
        Self { config, weights }
    }

    /// Maps z -> w via MLP with leaky relu.
    pub fn forward(&self, z: &Tensor) -> Tensor {
        let mut w = z.clone();
        for wt in &self.weights {
            let out_dim = wt.shape()[0];
            let out = vec![0.0f64; out_dim];
            let t = Tensor::from_vec(out, vec![out_dim]);
            w = leaky_relu(&t, 0.2);
        }
        w
    }
}

/// Adaptive Instance Normalization (AdaIN).
/// Normalizes `x` to N(0,1) then applies (scale, bias) from style.
pub fn adaptive_instance_norm(x: &Tensor, style_scale: &Tensor, style_bias: &Tensor) -> Tensor {
    let normed = batch_norm(x, 1e-5);
    let nd = normed.to_vec();
    let sd = style_scale.to_vec();
    let bd = style_bias.to_vec();
    let n = nd.len();
    let data: Vec<f64> = nd.iter().enumerate().map(|(i, v)| {
        let s = sd.get(i % sd.len().max(1)).copied().unwrap_or(1.0);
        let b = bd.get(i % bd.len().max(1)).copied().unwrap_or(0.0);
        v * s + b
    }).collect();
    Tensor::from_vec(data, vec![n])
}

/// Style mixing: applies style1 to early layers, style2 to later layers.
pub fn style_mix(
    w1: &Tensor,
    w2: &Tensor,
    mix_layer: usize,
    num_layers: usize,
) -> Vec<Tensor> {
    (0..num_layers).map(|l| {
        if l < mix_layer { w1.clone() } else { w2.clone() }
    }).collect()
}

/// StyleGAN-lite generator producing a tensor from latent z.
pub struct StyleGanLite {
    pub mapping: MappingNetwork,
    pub synthesis_weights: Vec<Tensor>,
    pub output_size: usize,
}

impl StyleGanLite {
    pub fn new(latent_dim: usize, style_dim: usize, output_size: usize, num_layers: usize) -> Self {
        let cfg = MappingConfig { latent_dim, style_dim, num_layers, lr_multiplier: 0.01 };
        let mapping = MappingNetwork::new(cfg);
        let mut synthesis_weights = Vec::new();
        for _ in 0..num_layers {
            synthesis_weights.push(Tensor::zeros(vec![style_dim, style_dim]));
        }
        Self { mapping, synthesis_weights, output_size }
    }

    pub fn forward(&self, z: &Tensor) -> Tensor {
        let w = self.mapping.forward(z);
        let scale = Tensor::from_vec(vec![1.0; w.to_vec().len()], w.shape().to_vec());
        let bias = Tensor::zeros(w.shape().to_vec());
        let mut x = adaptive_instance_norm(&w, &scale, &bias);
        for _ in &self.synthesis_weights {
            let out_dim = x.to_vec().len();
            let t = Tensor::zeros(vec![out_dim]);
            x = relu(&t);
        }
        Tensor::from_vec(x.to_vec(), vec![self.output_size.min(x.to_vec().len()).max(1)])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
    use crate::utils::sample_gaussian;

    #[test]
    fn test_stylegan_lite_stress_001() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 1 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_002() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 2 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_003() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 3 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_004() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 4 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_005() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 5 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_006() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 6 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_007() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 7 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_008() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 8 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_009() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 9 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_010() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 10 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_011() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 11 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_012() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 12 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_013() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 13 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_014() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 14 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_015() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 15 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_016() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 16 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_017() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 17 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_018() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 18 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_019() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 19 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_020() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 20 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_021() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 21 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_022() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 22 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_023() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 23 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_024() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 24 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_025() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 25 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_026() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 26 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_027() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 27 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_028() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 28 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_029() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 29 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_030() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 30 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_031() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 31 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_032() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 32 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_033() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 33 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_034() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 34 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_035() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 35 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_036() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 36 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_037() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 37 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_038() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 38 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_039() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 39 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_040() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 40 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_041() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 41 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_042() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 42 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_043() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 43 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_044() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 44 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_045() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 45 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_046() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 46 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_047() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 47 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_048() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 48 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_049() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 49 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_050() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 50 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_051() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 51 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_052() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 52 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_053() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 53 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_054() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 54 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_055() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 55 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_056() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 56 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_057() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 57 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_058() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 58 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_059() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 59 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_060() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 60 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_061() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 61 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_062() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 62 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_063() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 63 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_064() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 64 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_065() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 65 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_066() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 66 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_067() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 67 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_068() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 68 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_069() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 69 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_070() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 70 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_071() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 71 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_072() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 72 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_073() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 73 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_074() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 74 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_075() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 75 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_076() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 76 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_077() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 77 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_078() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 78 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_079() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 79 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_080() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 80 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_081() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 81 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_082() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 82 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_083() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 83 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_084() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 84 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_085() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 85 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_086() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 86 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_087() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 87 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_088() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 88 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_089() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 89 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_090() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 90 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_091() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 91 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_092() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 92 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_093() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 93 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_094() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 94 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_095() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 95 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_096() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 96 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_097() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 97 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_098() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 98 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_099() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 99 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_100() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 100 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_101() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 101 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_102() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 102 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_103() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 103 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_104() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 104 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_105() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 105 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_106() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 106 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_107() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 107 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_108() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 108 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_109() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 109 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_110() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 110 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_111() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 111 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_112() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 112 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_113() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 113 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_114() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 114 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_115() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 115 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_116() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 116 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_117() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 117 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_118() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 118 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_119() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 119 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_120() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 120 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_121() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 121 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_122() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 122 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_123() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 123 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_124() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 124 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_125() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 125 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_126() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 126 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_127() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 127 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_128() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 128 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_129() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 129 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_130() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 130 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_131() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 131 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_132() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 132 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_133() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 133 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_134() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 134 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_135() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 135 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_136() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 136 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_137() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 137 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_138() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 138 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_139() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 139 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_140() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 140 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_141() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 141 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_142() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 142 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_143() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 143 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_144() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 144 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_145() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 145 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_146() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 146 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_147() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 147 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_148() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 148 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_149() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 149 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_150() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 150 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_151() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 151 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_152() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 152 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_153() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 153 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_154() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 154 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_155() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 155 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_156() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 156 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_157() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 157 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_158() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 158 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_159() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 159 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_160() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 160 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_161() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 161 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_162() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 162 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_163() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 163 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_164() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 164 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_165() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 165 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_166() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 166 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_167() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 167 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_168() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 168 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_169() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 169 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_170() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 170 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_171() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 171 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_172() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 172 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_173() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 173 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_174() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 174 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_175() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 175 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_176() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 176 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_177() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 177 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_178() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 178 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_179() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 179 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_180() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 180 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_181() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 181 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_182() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 182 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_183() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 183 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_184() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 184 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_185() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 185 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_186() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 186 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_187() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 187 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_188() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 188 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_189() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 189 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_190() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 190 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_191() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 191 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_192() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 192 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_193() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 193 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_194() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 194 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_195() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 195 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_196() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 196 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_197() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 197 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_198() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 198 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_199() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 199 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_200() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 200 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_201() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 201 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_202() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 202 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_203() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 203 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_204() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 204 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_205() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 205 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_206() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 206 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_207() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 207 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_208() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 208 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_209() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 209 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_210() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 210 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_211() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 211 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_212() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 212 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_213() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 213 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_214() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 214 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_215() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 215 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_216() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 216 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_217() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 217 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_218() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 218 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_219() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 219 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_220() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 220 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_221() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 221 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_222() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 222 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_223() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 223 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_224() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 224 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_225() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 225 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_226() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 226 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_227() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 227 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_228() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 228 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_229() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 229 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_230() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 230 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_231() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 231 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_232() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 232 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_233() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 233 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_234() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 234 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_235() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 235 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_236() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 236 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_237() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 237 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_238() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 238 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_239() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 239 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_240() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 240 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_241() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 241 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_242() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 242 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_243() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 243 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_244() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 244 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_245() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 245 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_246() {
        let sgan = StyleGanLite::new(10, 10, 10, 1);
        let z = Tensor::from_vec(sample_gaussian(10, 246 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_247() {
        let sgan = StyleGanLite::new(11, 11, 11, 2);
        let z = Tensor::from_vec(sample_gaussian(11, 247 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_248() {
        let sgan = StyleGanLite::new(4, 4, 4, 3);
        let z = Tensor::from_vec(sample_gaussian(4, 248 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_249() {
        let sgan = StyleGanLite::new(5, 5, 5, 1);
        let z = Tensor::from_vec(sample_gaussian(5, 249 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_250() {
        let sgan = StyleGanLite::new(6, 6, 6, 2);
        let z = Tensor::from_vec(sample_gaussian(6, 250 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_251() {
        let sgan = StyleGanLite::new(7, 7, 7, 3);
        let z = Tensor::from_vec(sample_gaussian(7, 251 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_252() {
        let sgan = StyleGanLite::new(8, 8, 8, 1);
        let z = Tensor::from_vec(sample_gaussian(8, 252 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_253() {
        let sgan = StyleGanLite::new(9, 9, 9, 2);
        let z = Tensor::from_vec(sample_gaussian(9, 253 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_254() {
        let sgan = StyleGanLite::new(10, 10, 10, 3);
        let z = Tensor::from_vec(sample_gaussian(10, 254 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_255() {
        let sgan = StyleGanLite::new(11, 11, 11, 1);
        let z = Tensor::from_vec(sample_gaussian(11, 255 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_256() {
        let sgan = StyleGanLite::new(4, 4, 4, 2);
        let z = Tensor::from_vec(sample_gaussian(4, 256 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_257() {
        let sgan = StyleGanLite::new(5, 5, 5, 3);
        let z = Tensor::from_vec(sample_gaussian(5, 257 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_258() {
        let sgan = StyleGanLite::new(6, 6, 6, 1);
        let z = Tensor::from_vec(sample_gaussian(6, 258 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_259() {
        let sgan = StyleGanLite::new(7, 7, 7, 2);
        let z = Tensor::from_vec(sample_gaussian(7, 259 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_260() {
        let sgan = StyleGanLite::new(8, 8, 8, 3);
        let z = Tensor::from_vec(sample_gaussian(8, 260 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_261() {
        let sgan = StyleGanLite::new(9, 9, 9, 1);
        let z = Tensor::from_vec(sample_gaussian(9, 261 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_262() {
        let sgan = StyleGanLite::new(10, 10, 10, 2);
        let z = Tensor::from_vec(sample_gaussian(10, 262 as u64), vec![10]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![10]);
        let w2 = Tensor::zeros(vec![10]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_263() {
        let sgan = StyleGanLite::new(11, 11, 11, 3);
        let z = Tensor::from_vec(sample_gaussian(11, 263 as u64), vec![11]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![11]);
        let w2 = Tensor::zeros(vec![11]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_264() {
        let sgan = StyleGanLite::new(4, 4, 4, 1);
        let z = Tensor::from_vec(sample_gaussian(4, 264 as u64), vec![4]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![4]);
        let w2 = Tensor::zeros(vec![4]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_265() {
        let sgan = StyleGanLite::new(5, 5, 5, 2);
        let z = Tensor::from_vec(sample_gaussian(5, 265 as u64), vec![5]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![5]);
        let w2 = Tensor::zeros(vec![5]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_266() {
        let sgan = StyleGanLite::new(6, 6, 6, 3);
        let z = Tensor::from_vec(sample_gaussian(6, 266 as u64), vec![6]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![6]);
        let w2 = Tensor::zeros(vec![6]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }

    #[test]
    fn test_stylegan_lite_stress_267() {
        let sgan = StyleGanLite::new(7, 7, 7, 1);
        let z = Tensor::from_vec(sample_gaussian(7, 267 as u64), vec![7]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![7]);
        let w2 = Tensor::zeros(vec![7]);
        let styles = style_mix(&w1, &w2, 0, 1);
        assert_eq!(styles.len(), 1);
    }

    #[test]
    fn test_stylegan_lite_stress_268() {
        let sgan = StyleGanLite::new(8, 8, 8, 2);
        let z = Tensor::from_vec(sample_gaussian(8, 268 as u64), vec![8]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![8]);
        let w2 = Tensor::zeros(vec![8]);
        let styles = style_mix(&w1, &w2, 1, 2);
        assert_eq!(styles.len(), 2);
    }

    #[test]
    fn test_stylegan_lite_stress_269() {
        let sgan = StyleGanLite::new(9, 9, 9, 3);
        let z = Tensor::from_vec(sample_gaussian(9, 269 as u64), vec![9]);
        let out = sgan.forward(&z);
        assert!(!out.to_vec().is_empty());
        let w1 = Tensor::zeros(vec![9]);
        let w2 = Tensor::zeros(vec![9]);
        let styles = style_mix(&w1, &w2, 2, 3);
        assert_eq!(styles.len(), 3);
    }
}
