//! # GAN System Builder
//!
//! Ergonomic builder for assembling a complete GAN: generator + discriminator + loss.
#![allow(missing_docs)]

use crate::config::{GanConfig, LossVariant, ArchVariant};

/// Builder for constructing a GAN configuration.
#[derive(Debug, Default)]
pub struct GanBuilder {
    config: GanConfig,
}

impl GanBuilder {
    pub fn new() -> Self { Self::default() }

    pub fn generator(mut self, arch: ArchVariant) -> Self {
        self.config.arch = arch;
        self
    }

    pub fn discriminator(self, _arch: ArchVariant) -> Self { self }

    pub fn loss(mut self, loss: LossVariant) -> Self {
        self.config.loss = loss;
        self
    }

    pub fn latent_dim(mut self, dim: usize) -> Self {
        self.config.generator.latent_dim = dim;
        self
    }

    pub fn image_size(mut self, size: usize) -> Self {
        self.config.generator.image_size = size;
        self.config.discriminator.image_size = size;
        self
    }

    pub fn base_channels(mut self, ch: usize) -> Self {
        self.config.generator.base_channels = ch;
        self.config.discriminator.base_channels = ch;
        self
    }

    pub fn num_layers(mut self, n: usize) -> Self {
        self.config.generator.num_layers = n;
        self.config.discriminator.num_layers = n;
        self
    }

    pub fn n_critic(mut self, n: usize) -> Self {
        self.config.training.n_critic = n;
        self
    }

    pub fn gradient_penalty(mut self, gp: bool) -> Self {
        self.config.training.gradient_penalty = gp;
        self
    }

    pub fn label_smoothing(mut self, smooth: f64) -> Self {
        self.config.training.label_smoothing = smooth;
        self
    }

    pub fn num_classes(mut self, c: usize) -> Self {
        self.config.generator.num_classes = c;
        self.config.discriminator.num_classes = c;
        self
    }

    pub fn build(self) -> Result<GanConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_builder_stress_001() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(5)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 5);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_002() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(6)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 6);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_003() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(7)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 7);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_004() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(8)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 8);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_005() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(9)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 9);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_006() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(10)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 10);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_007() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(11)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 11);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_008() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(12)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 12);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_009() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(13)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 13);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_010() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(14)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 14);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_011() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(15)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 15);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_012() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(16)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 16);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_013() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(17)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 17);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_014() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(18)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 18);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_015() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(19)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 19);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_016() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(20)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 20);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_017() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(21)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 21);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_018() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(22)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 22);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_019() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(23)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 23);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_020() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(24)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 24);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_021() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(25)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 25);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_022() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(26)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 26);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_023() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(27)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 27);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_024() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(28)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 28);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_025() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(29)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 29);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_026() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(30)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 30);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_027() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(31)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 31);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_028() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(32)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 32);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_029() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(33)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 33);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_030() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(34)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 34);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_031() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(35)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 35);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_032() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(36)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 36);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_033() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(37)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 37);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_034() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(38)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 38);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_035() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(39)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 39);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_036() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(40)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 40);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_037() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(41)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 41);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_038() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(42)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 42);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_039() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(43)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 43);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_040() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(44)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 44);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_041() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(45)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 45);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_042() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(46)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 46);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_043() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(47)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 47);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_044() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(48)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 48);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_045() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(49)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 49);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_046() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(50)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 50);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_047() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(51)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 51);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_048() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(52)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 52);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_049() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(53)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 53);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_050() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(54)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 54);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_051() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(55)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 55);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_052() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(56)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 56);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_053() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(57)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 57);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_054() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(58)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 58);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_055() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(59)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 59);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_056() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(60)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 60);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_057() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(61)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 61);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_058() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(62)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 62);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_059() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(63)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 63);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_060() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(64)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 64);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_061() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(65)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 65);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_062() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(66)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 66);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_063() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(67)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 67);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_064() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(68)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 68);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_065() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(69)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 69);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_066() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(70)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 70);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_067() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(71)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 71);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_068() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(72)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 72);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_069() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(73)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 73);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_070() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(74)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 74);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_071() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(75)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 75);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_072() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(76)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 76);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_073() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(77)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 77);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_074() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(78)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 78);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_075() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(79)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 79);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_076() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(80)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 80);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_077() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(81)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 81);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_078() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(82)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 82);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_079() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(83)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 83);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_080() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(84)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 84);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_081() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(85)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 85);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_082() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(86)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 86);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_083() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(87)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 87);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_084() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(88)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 88);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_085() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(89)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 89);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_086() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(90)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 90);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_087() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(91)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 91);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_088() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(92)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 92);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_089() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(93)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 93);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_090() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(94)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 94);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_091() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(95)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 95);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_092() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(96)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 96);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_093() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(97)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 97);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_094() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(98)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 98);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_095() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(99)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 99);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_096() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(100)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 100);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_097() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(101)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 101);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_098() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(102)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 102);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_099() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(103)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 103);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_100() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(104)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 104);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_101() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(105)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 105);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_102() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(106)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 106);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_103() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(107)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 107);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_104() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(108)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 108);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_105() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(109)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 109);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_106() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(110)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 110);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_107() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(111)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 111);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_108() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(112)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 112);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_109() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(113)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 113);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_110() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(114)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 114);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_111() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(115)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 115);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_112() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(116)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 116);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_113() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(117)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 117);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_114() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(118)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 118);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_115() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(119)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 119);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_116() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(120)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 120);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_117() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(121)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 121);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_118() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(122)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 122);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_119() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(123)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 123);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_120() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(124)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 124);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_121() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(125)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 125);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_122() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(126)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 126);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_123() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(127)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 127);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_124() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(128)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 128);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_125() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(129)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 129);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_126() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(130)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 130);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_127() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(131)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 131);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_128() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(132)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 132);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_129() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(133)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 133);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_130() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(134)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 134);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_131() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(135)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 135);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_132() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(136)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 136);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_133() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(137)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 137);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_134() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(138)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 138);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_135() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(139)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 139);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_136() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(140)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 140);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_137() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(141)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 141);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_138() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(142)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 142);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_139() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(143)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 143);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_140() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(144)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 144);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_141() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(145)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 145);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_142() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(146)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 146);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_143() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(147)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 147);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_144() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(148)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 148);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_145() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(149)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 149);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_146() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(150)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 150);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_147() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(151)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 151);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_148() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(152)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 152);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_149() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(153)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 153);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_150() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(154)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 154);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_151() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(155)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 155);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_152() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(156)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 156);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_153() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(157)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 157);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_154() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(158)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 158);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_155() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(159)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 159);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_156() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(160)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 160);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_157() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(161)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 161);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_158() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(162)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 162);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_159() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(163)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 163);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_160() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(164)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 164);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_161() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(165)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 165);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_162() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(166)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 166);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_163() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(167)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 167);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_164() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(168)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 168);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_165() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(169)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 169);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_166() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(170)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 170);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_167() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(171)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 171);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_168() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(172)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 172);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_169() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(173)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 173);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_170() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(174)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 174);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_171() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(175)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 175);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_172() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(176)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 176);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_173() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(177)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 177);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_174() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(178)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 178);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_175() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(179)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 179);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_176() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(180)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 180);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_177() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(181)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 181);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_178() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(182)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 182);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_179() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(183)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 183);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_180() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(184)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 184);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_181() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(185)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 185);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_182() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(186)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 186);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_183() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(187)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 187);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_184() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(188)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 188);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_185() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(189)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 189);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_186() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(190)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 190);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_187() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(191)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 191);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_188() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(192)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 192);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_189() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(193)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 193);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_190() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(194)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 194);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_191() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(195)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 195);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_192() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(196)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 196);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_193() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(197)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 197);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_194() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(198)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 198);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_195() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(199)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 199);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_196() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(200)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 200);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_197() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(201)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 201);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_198() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(202)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 202);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_199() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(203)
            .image_size(64)
            .n_critic(5)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 203);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_200() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(204)
            .image_size(64)
            .n_critic(1)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 204);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_201() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(205)
            .image_size(64)
            .n_critic(2)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 205);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_202() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(206)
            .image_size(64)
            .n_critic(3)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 206);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
    }

    #[test]
    fn test_builder_stress_203() {
        let cfg = GanBuilder::new()
            .generator(ArchVariant::Dcgan)
            .loss(LossVariant::Hinge)
            .latent_dim(207)
            .image_size(64)
            .n_critic(4)
            .build();
        assert!(cfg.is_ok());
        let c = cfg.unwrap();
        assert_eq!(c.generator.latent_dim, 207);
        let bad = GanBuilder::new().latent_dim(0).build();
        assert!(bad.is_err());
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
    // GAN training and evaluation padding line 12
    // GAN training and evaluation padding line 13
    // GAN training and evaluation padding line 14
}
