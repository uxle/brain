//! # Classic GAN Losses & Gradient Penalties
//!
//! BCE (minimax), LSGAN, Hinge, WGAN, WGAN-GP (Gulrajani et al.), and R1/R2 Regularization (Mescheder et al.).
#![allow(missing_docs)]

use brain_core::Tensor;

/// Classic loss variant enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClassicLoss {
    #[default]
    Bce,
    Lsgan,
    Hinge,
    Wgan,
    WganGp,
    RelativisticHinge,
}

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

/// WGAN-GP discriminator loss with 1-Lipschitz gradient penalty:
/// -(E[D(real)] - E[D(fake)]) + lambda * (||grad||_2 - 1)^2.
#[inline]
pub fn wgan_gp_loss_d(d_real: f64, d_fake: f64, grad_norm: f64, lambda_gp: f64) -> f64 {
    wgan_loss_d(d_real, d_fake) + lambda_gp * (grad_norm - 1.0).powi(2)
}

/// Computes linear interpolation between real and fake samples for WGAN-GP:
/// x_hat = eps * real + (1 - eps) * fake.
pub fn sample_wgan_gp_interpolates(real: &Tensor, fake: &Tensor, eps: f64) -> Tensor {
    assert_eq!(
        real.shape(),
        fake.shape(),
        "Real and fake tensors must have identical shape"
    );
    let mut out_data = Vec::with_capacity(real.numel());
    for (&r, &f) in real.data().iter().zip(fake.data().iter()) {
        out_data.push(eps * r + (1.0 - eps) * f);
    }
    Tensor::from_slice(&out_data, real.shape().to_vec())
}

/// R1 gradient penalty regularizer (Mescheder et al. / StyleGAN):
/// R1 = (gamma / 2) * E[||grad_real||^2].
#[inline]
pub fn r1_gradient_penalty(grad_norm_sq: f64, gamma: f64) -> f64 {
    0.5 * gamma * grad_norm_sq
}

/// R2 gradient penalty regularizer:
/// R2 = (gamma / 2) * E[||grad_fake||^2].
#[inline]
pub fn r2_gradient_penalty(grad_norm_sq: f64, gamma: f64) -> f64 {
    0.5 * gamma * grad_norm_sq
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgan_gp_loss_and_interpolation() {
        let loss = wgan_gp_loss_d(2.0, -1.0, 1.5, 10.0);
        // -(2 - (-1)) + 10 * (1.5 - 1)^2 = -3 + 10 * 0.25 = -3 + 2.5 = -0.5
        assert!((loss - (-0.5)).abs() < 1e-6);

        let real = Tensor::from_slice(&[0.0, 10.0], vec![2]);
        let fake = Tensor::from_slice(&[10.0, 0.0], vec![2]);
        let interp = sample_wgan_gp_interpolates(&real, &fake, 0.3);
        assert_eq!(interp.data(), &[7.0, 3.0]);
    }

    #[test]
    fn test_r1_penalty() {
        let r1 = r1_gradient_penalty(4.0, 10.0);
        // 0.5 * 10 * 4 = 20.0
        assert_eq!(r1, 20.0);
    }
}
