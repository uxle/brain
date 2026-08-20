//! Tests for GAN architectures and losses
use brain_gan::*;
use brain_core::Tensor;

#[test]
fn test_gan_config_and_builder() {
    let cfg = GanConfig::default();
    assert_eq!(cfg.generator.latent_dim, 128);

    let b = GanBuilder::new().latent_dim(64).build();
    assert!(b.is_ok());
    assert_eq!(b.unwrap().generator.latent_dim, 64);
}

#[test]
fn test_dcgan_generator_and_discriminator_shapes() {
    let gen_cfg = GeneratorConfig {
        latent_dim: 10,
        base_channels: 8,
        num_layers: 2,
        image_size: 16,
        output_channels: 1,
        latent_type: LatentType::Gaussian,
        output_activation: OutputActivation::Tanh,
        num_classes: 0,
    };
    let gen = DcganGenerator::new(gen_cfg);
    let z = Tensor::from_slice(&[0.1; 10], vec![1, 10]);
    let fake = gen.forward(&z);
    assert!(!fake.shape().is_empty());
}
