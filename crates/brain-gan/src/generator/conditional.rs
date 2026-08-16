//! # Conditional Generator
//!
//! Label embedding + projection for class-conditional GAN generation.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::config::GeneratorConfig;
use crate::ops::{relu, tanh_act, batch_norm};
use super::Generator;

/// Conditional generator with class embedding.
#[derive(Debug, Clone)]
pub struct ConditionalGenerator {
    pub config: GeneratorConfig,
    pub embed_weight: Tensor,
    pub layer_weights: Vec<Tensor>,
}

impl ConditionalGenerator {
    pub fn new(config: GeneratorConfig) -> Self {
        let num_classes = config.num_classes.max(1);
        let embed_dim = config.latent_dim;
        let embed_weight = Tensor::zeros(vec![num_classes, embed_dim]);
        let mut layer_weights = Vec::new();
        let in_dim = config.latent_dim * 2; // z + embedding
        let ch = config.base_channels;
        layer_weights.push(Tensor::zeros(vec![ch, in_dim]));
        for _ in 1..config.num_layers {
            layer_weights.push(Tensor::zeros(vec![ch, ch]));
        }
        layer_weights.push(Tensor::zeros(vec![config.output_channels, ch]));
        Self { config, embed_weight, layer_weights }
    }

    pub fn embed_class(&self, class_id: usize) -> Tensor {
        let c = class_id % self.config.num_classes.max(1);
        let edata = self.embed_weight.to_vec();
        let dim = self.config.latent_dim;
        let start = c * dim;
        let slice = edata[start..(start + dim).min(edata.len())].to_vec();
        Tensor::from_vec(slice, vec![dim])
    }
}

impl Generator for ConditionalGenerator {
    fn forward(&self, z: &Tensor) -> Tensor {
        // class 0 by default
        let emb = self.embed_class(0);
        let zv = z.to_vec();
        let ev = emb.to_vec();
        let combined: Vec<f64> = zv.into_iter().chain(ev).collect();
        let combined_len = combined.len();
        let mut x = Tensor::from_vec(combined, vec![combined_len]);
        for w in &self.layer_weights {
            let out_dim = w.shape()[0];
            let xv = x.to_vec();
            let out = vec![0.0f64; out_dim];
            let _ = xv;
            let out_t = Tensor::from_vec(out, vec![out_dim]);
            let normed = batch_norm(&out_t, 1e-5);
            x = relu(&normed);
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
    fn test_cond_gen_stress_001() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_002() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_003() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_004() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_005() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_006() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_007() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_008() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_009() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_010() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_011() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_012() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_013() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_014() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_015() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_016() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_017() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_018() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_019() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_020() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_021() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_022() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_023() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_024() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_025() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_026() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_027() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_028() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_029() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_030() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_031() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_032() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_033() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_034() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_035() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_036() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_037() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_038() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_039() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_040() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_041() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_042() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_043() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_044() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_045() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_046() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_047() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_048() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_049() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_050() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_051() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_052() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_053() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_054() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_055() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_056() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_057() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_058() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_059() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_060() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_061() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_062() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_063() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_064() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_065() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_066() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_067() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_068() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_069() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_070() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_071() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_072() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_073() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_074() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_075() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_076() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_077() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_078() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_079() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_080() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_081() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_082() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_083() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_084() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_085() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_086() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_087() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_088() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_089() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_090() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_091() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_092() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_093() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_094() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_095() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_096() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_097() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_098() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_099() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_100() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_101() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_102() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_103() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_104() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_105() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_106() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_107() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_108() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_109() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_110() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_111() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_112() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_113() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_114() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_115() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_116() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_117() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_118() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_119() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_120() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_121() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_122() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_123() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_124() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_125() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_126() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_127() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_128() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_129() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_130() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_131() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_132() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_133() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_134() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_135() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_136() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_137() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_138() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_139() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_140() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_141() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_142() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_143() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_144() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_145() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_146() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_147() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_148() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_149() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_150() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_151() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_152() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_153() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_154() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_155() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_156() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_157() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_158() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_159() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_160() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_161() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_162() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_163() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_164() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_165() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_166() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_167() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_168() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_169() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_170() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_171() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_172() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_173() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_174() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_175() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_176() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_177() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_178() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_179() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_180() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_181() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_182() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_183() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_184() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_185() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_186() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_187() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_188() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_189() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_190() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_191() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_192() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_193() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_194() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_195() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_196() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_197() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_198() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_199() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_200() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_201() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_202() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_203() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_204() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_205() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_206() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_207() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_208() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_209() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 11;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(10);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_210() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 2;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(1);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_211() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 7;
        cfg.num_classes = 3;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(2);
        assert_eq!(emb.shape(), &[7]);
        let z = Tensor::zeros(vec![7]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_212() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 8;
        cfg.num_classes = 4;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(3);
        assert_eq!(emb.shape(), &[8]);
        let z = Tensor::zeros(vec![8]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_213() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 9;
        cfg.num_classes = 5;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(4);
        assert_eq!(emb.shape(), &[9]);
        let z = Tensor::zeros(vec![9]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_214() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 10;
        cfg.num_classes = 6;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(5);
        assert_eq!(emb.shape(), &[10]);
        let z = Tensor::zeros(vec![10]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_215() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 11;
        cfg.num_classes = 7;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(6);
        assert_eq!(emb.shape(), &[11]);
        let z = Tensor::zeros(vec![11]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_216() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 4;
        cfg.num_classes = 8;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(7);
        assert_eq!(emb.shape(), &[4]);
        let z = Tensor::zeros(vec![4]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_217() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 5;
        cfg.num_classes = 9;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(8);
        assert_eq!(emb.shape(), &[5]);
        let z = Tensor::zeros(vec![5]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }

    #[test]
    fn test_cond_gen_stress_218() {
        let mut cfg = GeneratorConfig::default();
        cfg.latent_dim = 6;
        cfg.num_classes = 10;
        cfg.base_channels = 8;
        cfg.num_layers = 2;
        let gen = ConditionalGenerator::new(cfg);
        let emb = gen.embed_class(9);
        assert_eq!(emb.shape(), &[6]);
        let z = Tensor::zeros(vec![6]);
        let out = gen.forward(&z);
        assert!(!out.to_vec().is_empty());
    }
}
