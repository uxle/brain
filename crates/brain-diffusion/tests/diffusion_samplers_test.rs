//! Tests for diffusion schedules and models
use brain_diffusion::{
    CosineSchedule, DiffusionConfig, LinearSchedule, NoiseSchedule, Unet2d, UnetConfig,
};

#[test]
fn test_linear_and_cosine_noise_schedules() {
    let lin = LinearSchedule::new(100, 0.0001, 0.02);
    assert_eq!(lin.timesteps(), 100);
    assert!(lin.beta(0) < lin.beta(99));
    assert!(lin.alpha_cumprod(0) > lin.alpha_cumprod(99));

    let cos = CosineSchedule::new(100, 0.008);
    assert_eq!(cos.timesteps(), 100);
    assert!(cos.alpha_cumprod(0) > cos.alpha_cumprod(99));
}

#[test]
fn test_diffusion_config_and_unet_creation() {
    let cfg = DiffusionConfig::default();
    assert_eq!(cfg.timesteps, 1000);

    let unet_cfg = UnetConfig {
        in_channels: 3,
        out_channels: 3,
        model_channels: 16,
        num_res_blocks: 1,
    };
    let unet = Unet2d::new(unet_cfg);
    assert_eq!(unet.config.in_channels, 3);
}
