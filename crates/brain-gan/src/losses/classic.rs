//! # Classic GAN Losses
//!
//! BCE (minimax), LSGAN, Hinge, WGAN, Relativistic — all scalar-domain implementations.
#![allow(missing_docs)]

/// Hinge discriminator loss: max(0, 1-d_real) + max(0, 1+d_fake).
#[inline]
pub fn hinge_loss_d(d_real: f64, d_fake: f64) -> f64 {
    (1.0 - d_real).max(0.0) + (1.0 + d_fake).max(0.0)
}

/// Hinge generator loss: -E[D(fake)].
#[inline]
pub fn hinge_loss_g(d_fake: f64) -> f64 {
    -d_fake
}

/// WGAN discriminator loss: -(E[D(real)] - E[D(fake)]).
#[inline]
pub fn wgan_loss_d(d_real: f64, d_fake: f64) -> f64 {
    -(d_real - d_fake)
}

/// WGAN generator loss: -E[D(fake)].
#[inline]
pub fn wgan_loss_g(d_fake: f64) -> f64 {
    -d_fake
}

/// LSGAN discriminator loss: 0.5*((d_real-1)^2 + d_fake^2).
#[inline]
pub fn lsgan_loss_d(d_real: f64, d_fake: f64) -> f64 {
    0.5 * ((d_real - 1.0).powi(2) + d_fake.powi(2))
}

/// LSGAN generator loss: 0.5*(d_fake-1)^2.
#[inline]
pub fn lsgan_loss_g(d_fake: f64) -> f64 {
    0.5 * (d_fake - 1.0).powi(2)
}

/// BCE discriminator loss with optional label smoothing.
pub fn bce_loss_d(d_real: f64, d_fake: f64, label_smooth: f64) -> f64 {
    let real_label = 1.0 - label_smooth;
    let p_r = d_real.clamp(1e-7, 1.0 - 1e-7);
    let p_f = d_fake.clamp(1e-7, 1.0 - 1e-7);
    let loss_real = -(real_label * p_r.ln() + (1.0 - real_label) * (1.0 - p_r).ln());
    let loss_fake = -(0.0 * p_f.ln() + 1.0 * (1.0 - p_f).ln());
    (loss_real + loss_fake) * 0.5
}

/// BCE generator loss: -log(D(fake)).
pub fn bce_loss_g(d_fake: f64) -> f64 {
    let p = d_fake.clamp(1e-7, 1.0 - 1e-7);
    -p.ln()
}

/// Relativistic discriminator loss (RaGAN): hinge variant.
pub fn ragan_loss_d(d_real: f64, d_fake: f64) -> f64 {
    let rel_real = d_real - d_fake;
    let rel_fake = d_fake - d_real;
    (1.0 - rel_real).max(0.0) + (1.0 + rel_fake).max(0.0)
}

/// Relativistic generator loss.
pub fn ragan_loss_g(d_real: f64, d_fake: f64) -> f64 {
    let rel_real = d_real - d_fake;
    let rel_fake = d_fake - d_real;
    (1.0 + rel_real).max(0.0) + (1.0 - rel_fake).max(0.0)
}

/// Struct dispatch for classic GAN loss variants.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassicLoss;

impl ClassicLoss {
    pub fn new() -> Self {
        Self
    }

    pub fn d_loss(&self, d_real: f64, d_fake: f64, variant: &str, smooth: f64) -> f64 {
        match variant {
            "hinge" => hinge_loss_d(d_real, d_fake),
            "wgan" => wgan_loss_d(d_real, d_fake),
            "lsgan" => lsgan_loss_d(d_real, d_fake),
            "ragan" => ragan_loss_d(d_real, d_fake),
            _ => bce_loss_d(d_real, d_fake, smooth),
        }
    }

    pub fn g_loss(&self, d_fake: f64, d_real: f64, variant: &str) -> f64 {
        match variant {
            "hinge" => hinge_loss_g(d_fake),
            "wgan" => wgan_loss_g(d_fake),
            "lsgan" => lsgan_loss_g(d_fake),
            "ragan" => ragan_loss_g(d_real, d_fake),
            _ => bce_loss_g(d_fake),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
