//! Tests for GAN architectures and losses
use brain_core::Tensor;
use brain_gan::*;

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

#[test]
fn test_wgan_gp_and_r1_penalties() {
    let loss_d = wgan_gp_loss_d(1.5, -0.5, 1.2, 10.0);
    // -(1.5 - (-0.5)) + 10.0 * (1.2 - 1.0)^2 = -2.0 + 10.0 * 0.04 = -2.0 + 0.4 = -1.6
    assert!((loss_d - (-1.6)).abs() < 1e-6);

    let r1 = r1_gradient_penalty(2.0, 10.0);
    // 0.5 * 10.0 * 2.0 = 10.0
    assert_eq!(r1, 10.0);

    let real = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
    let fake = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
    let interp = sample_wgan_gp_interpolates(&real, &fake, 0.5);
    assert_eq!(interp.data(), &[2.5, 3.5, 4.5]);
}

#[test]
fn test_spectral_norm_power_iteration() {
    let w = Tensor::from_slice(&[3.0, 0.0, 0.0, 4.0], vec![2, 2]);
    let u = Tensor::from_slice(&[1.0, 0.0], vec![2]);

    let (w_sn, u_next, sigma) = spectral_norm_apply(&w, &u);
    assert!(sigma > 0.0);
    assert_eq!(w_sn.shape(), &[2, 2]);
    assert_eq!(u_next.shape(), &[2]);
}
