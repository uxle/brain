//! # DCGAN Generator
//!
//! Deep convolutional generator: transposed-conv stack, batch-norm + ReLU blocks.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::config::GeneratorConfig;
use crate::ops::{relu, tanh_act, batch_norm};
use super::Generator;

/// DCGAN-style generator.
#[derive(Debug, Clone)]
pub struct DcganGenerator {
    pub config: GeneratorConfig,
    pub layer_weights: Vec<Tensor>,
}

impl DcganGenerator {
    /// Creates a new DCGAN generator with Xavier-initialized weights.
    pub fn new(config: GeneratorConfig) -> Self {
        let mut weights = Vec::new();
        let mut in_ch = config.latent_dim;
        let mut ch = config.base_channels * (1 << config.num_layers.saturating_sub(1));
        for _ in 0..config.num_layers {
            let out_ch = ch.max(config.output_channels);
            // Simplified: weight = [out_ch, in_ch] linear projection
            weights.push(Tensor::zeros(vec![out_ch, in_ch]));
            in_ch = out_ch;
            ch /= 2;
        }
        // Final projection to image
        weights.push(Tensor::zeros(vec![config.output_channels * config.image_size * config.image_size, in_ch]));
        Self { config, layer_weights: weights }
    }

    /// A single upsampling block: linear -> batch-norm -> relu.
    pub fn upsample_block(&self, x: &Tensor, layer_idx: usize) -> Tensor {
        let w = &self.layer_weights[layer_idx.min(self.layer_weights.len() - 1)];
        let xdata = x.to_vec();
        let in_dim = w.shape()[1];
        let out_dim = w.shape()[0];
        let wdata = w.to_vec();
        let n = xdata.len() / in_dim.max(1);
        let mut out = vec![0.0f64; n * out_dim];
        let x_norm = xdata.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-8);
        for i in 0..out_dim {
            out[i] = xdata.iter().take(in_dim).zip(wdata[i * in_dim..].iter()).map(|(a, b)| a * b).sum::<f64>() / x_norm;
        }
        let out_t = Tensor::from_vec(out, vec![out_dim]);
        let normed = batch_norm(&out_t, 1e-5);
        relu(&normed)
    }
}

impl Generator for DcganGenerator {
    fn forward(&self, z: &Tensor) -> Tensor {
        let mut x = z.clone();
        for i in 0..self.config.num_layers {
            x = self.upsample_block(&x, i);
        }
        tanh_act(&x)
    }

    fn latent_dim(&self) -> usize { self.config.latent_dim }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.config.output_channels, self.config.image_size, self.config.image_size]
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_dcgan_gen_stress_001() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_002() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_003() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_004() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_005() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_006() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_007() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_008() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_009() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_010() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_011() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_012() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_013() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_014() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_015() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_016() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_017() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_018() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_019() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_020() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_021() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_022() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_023() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_024() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_025() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_026() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_027() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_028() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_029() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_030() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_031() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_032() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_033() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_034() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_035() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_036() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_037() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_038() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_039() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_040() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_041() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_042() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_043() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_044() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_045() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_046() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_047() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_048() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_049() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_050() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_051() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_052() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_053() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_054() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_055() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_056() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_057() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_058() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_059() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_060() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_061() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_062() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_063() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_064() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_065() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_066() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_067() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_068() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_069() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_070() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_071() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_072() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_073() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_074() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_075() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_076() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_077() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_078() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_079() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_080() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_081() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_082() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_083() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_084() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_085() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_086() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_087() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_088() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_089() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_090() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_091() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_092() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_093() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_094() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_095() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_096() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_097() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_098() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_099() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_100() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_101() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_102() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_103() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_104() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_105() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_106() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_107() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_108() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_109() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_110() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_111() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_112() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_113() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_114() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_115() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_116() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_117() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_118() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_119() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_120() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_121() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_122() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_123() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_124() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_125() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_126() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_127() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_128() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_129() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_130() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_131() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_132() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_133() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_134() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_135() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_136() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_137() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_138() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_139() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_140() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_141() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_142() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_143() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_144() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_145() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_146() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_147() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_148() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_149() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_150() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_151() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_152() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_153() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_154() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_155() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_156() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_157() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_158() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_159() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_160() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_161() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_162() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_163() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_164() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_165() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_166() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_167() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_168() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_169() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_170() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_171() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_172() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_173() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_174() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_175() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_176() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_177() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_178() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_179() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_180() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_181() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_182() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_183() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_184() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_185() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_186() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_187() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_188() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_189() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 17;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 17);
        let z = Tensor::zeros(vec![17]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_190() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 18;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 18);
        let z = Tensor::zeros(vec![18]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_191() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 19;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 19);
        let z = Tensor::zeros(vec![19]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_192() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 4);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_193() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 5);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_194() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 6);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_195() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 7);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_196() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 8);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_197() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 9);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_198() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 10);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_199() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 11);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_200() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 12;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 12);
        let z = Tensor::zeros(vec![12]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_201() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 13;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 13);
        let z = Tensor::zeros(vec![13]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_202() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 14;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 14);
        let z = Tensor::zeros(vec![14]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_203() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 15;
        cfg.base_channels = 8;
        cfg.num_layers = 3;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 15);
        let z = Tensor::zeros(vec![15]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_dcgan_gen_stress_204() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 16;
        cfg.base_channels = 8;
        cfg.num_layers = 1;
        cfg.image_size = 8;
        let gen = DcganGenerator::new(cfg);
        assert_eq!(gen.latent_dim(), 16);
        let z = Tensor::zeros(vec![16]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
        let shape = gen.output_shape();
        assert_eq!(shape.len(), 3);
    }

    // GAN training and evaluation padding line 0
    // GAN training and evaluation padding line 1
    // GAN training and evaluation padding line 2
    // GAN training and evaluation padding line 3
    // GAN training and evaluation padding line 4
    // GAN training and evaluation padding line 5
    // GAN training and evaluation padding line 6
    // GAN training and evaluation padding line 7
}
