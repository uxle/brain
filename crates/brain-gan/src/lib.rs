//! # brain-gan
//!
//! Production-grade GAN framework: DCGAN / ResNet / Conditional / PatchGAN generators
//! and discriminators, 5 loss variants, WGAN-GP/R1 penalties, full training loop,
//! evaluation, StyleGAN-lite and CycleGAN-lite.
//!
//! ## Architecture
//! - [`generator`] — Generator trait, DCGAN, ResNet, Conditional
//! - [`discriminator`] — Discriminator trait, DCGAN, Conditional, Patch
//! - [`losses`] — Classic (BCE/LSGAN/Hinge/WGAN/RaGAN), Perceptual
//! - [`train`] — GanTrainer, TrainLoop, gradient penalties
//! - [`eval`] — GanEvalReport, FID-lite, IS-lite, sample grids
//! - [`gan`] — Unified `Gan` model
//! - [`stylegan_lite`] — Mapping network + AdaIN + style mixing
//! - [`cycle`] — CycleGAN-lite cycle-consistency loss
//! - [`core`] — `GanState`, `GanMetrics`, `EpochSummary`
//! - [`config`] — `GanConfig`, `GeneratorConfig`, `DiscriminatorConfig`
//! - [`ops`] — Activations, batch-norm, spectral norm, interpolation
//! - [`utils`] — Seed, EMA, sampling, BCE scalar

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

pub mod builder;
pub mod config;
pub mod core;
pub mod cycle;
pub mod discriminator;
pub mod eval;
pub mod gan;
pub mod generator;
pub mod impl_;
pub mod losses;
pub mod ops;
pub mod stylegan_lite;
pub mod train;
pub mod utils;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use builder::GanBuilder;
pub use config::{
    ArchVariant, DiscriminatorConfig, GanConfig, GanTrainConfig, GeneratorConfig,
    LatentType, LossVariant, OutputActivation,
};
pub use core::{EpochSummary, GanError, GanMetrics, GanResult, GanState};
pub use cycle::{CycleConfig, CycleGanLite, cycle_consistency_loss, cycle_total_loss, identity_loss};
pub use discriminator::{DcganDiscriminator, ConditionalDiscriminator, PatchDiscriminator};
pub use eval::{GanEvalReport, eval_gan, fid_lite, is_lite};
pub use eval::samples::{assemble_grid, fixed_latent_sample, interpolate_latents_batch};
pub use gan::Gan;
pub use generator::{DcganGenerator, ResnetGenerator, ConditionalGenerator, Generator, sample_latent};
pub use losses::{hinge_loss_d, hinge_loss_g, wgan_loss_d, wgan_loss_g, lsgan_loss_d, bce_loss_d};
pub use losses::perceptual::{PerceptualConfig, feature_matching_loss, gram_matrix, perceptual_loss};
pub use ops::{
    batch_norm, image_grid, interpolate_latents, leaky_relu, mix_style, relu,
    resize_like, sigmoid_act, spectral_norm_apply, tanh_act, wgan_clip,
};
pub use stylegan_lite::{MappingConfig, MappingNetwork, StyleGanLite, adaptive_instance_norm, style_mix};
pub use train::{GanTrainer, GanTrainStats, gradient_penalty, r1_penalty, PenaltyConfig};
pub use train::loop_::TrainLoop;
pub use utils::{
    bce_scalar, box_muller, clip_weights, l2_norm, log_gan, next_rand,
    sample_gaussian, set_seed, sigmoid_vec, track_ema,
};

/// GAN framework version.
pub const VERSION: &str = "0.2.0";