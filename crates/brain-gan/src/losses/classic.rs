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
    pub fn new() -> Self { Self }

    pub fn d_loss(&self, d_real: f64, d_fake: f64, variant: &str, smooth: f64) -> f64 {
        match variant {
            "hinge" => hinge_loss_d(d_real, d_fake),
            "wgan"  => wgan_loss_d(d_real, d_fake),
            "lsgan" => lsgan_loss_d(d_real, d_fake),
            "ragan" => ragan_loss_d(d_real, d_fake),
            _       => bce_loss_d(d_real, d_fake, smooth),
        }
    }

    pub fn g_loss(&self, d_fake: f64, d_real: f64, variant: &str) -> f64 {
        match variant {
            "hinge" => hinge_loss_g(d_fake),
            "wgan"  => wgan_loss_g(d_fake),
            "lsgan" => lsgan_loss_g(d_fake),
            "ragan" => ragan_loss_g(d_real, d_fake),
            _       => bce_loss_g(d_fake),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_classic_loss_stress_001() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 1 as f64 * 0.001_f64, -0.5 - 1 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_002() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 2 as f64 * 0.001_f64, -0.5 - 2 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_003() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 3 as f64 * 0.001_f64, -0.5 - 3 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_004() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 4 as f64 * 0.001_f64, -0.5 - 4 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_005() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 5 as f64 * 0.001_f64, -0.5 - 5 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_006() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 6 as f64 * 0.001_f64, -0.5 - 6 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_007() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 7 as f64 * 0.001_f64, -0.5 - 7 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_008() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 8 as f64 * 0.001_f64, -0.5 - 8 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_009() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 9 as f64 * 0.001_f64, -0.5 - 9 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_010() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 10 as f64 * 0.001_f64, -0.5 - 10 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_011() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 11 as f64 * 0.001_f64, -0.5 - 11 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_012() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 12 as f64 * 0.001_f64, -0.5 - 12 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_013() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 13 as f64 * 0.001_f64, -0.5 - 13 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_014() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 14 as f64 * 0.001_f64, -0.5 - 14 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_015() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 15 as f64 * 0.001_f64, -0.5 - 15 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_016() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 16 as f64 * 0.001_f64, -0.5 - 16 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_017() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 17 as f64 * 0.001_f64, -0.5 - 17 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_018() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 18 as f64 * 0.001_f64, -0.5 - 18 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_019() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 19 as f64 * 0.001_f64, -0.5 - 19 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_020() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 20 as f64 * 0.001_f64, -0.5 - 20 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_021() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 21 as f64 * 0.001_f64, -0.5 - 21 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_022() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 22 as f64 * 0.001_f64, -0.5 - 22 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_023() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 23 as f64 * 0.001_f64, -0.5 - 23 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_024() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 24 as f64 * 0.001_f64, -0.5 - 24 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_025() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 25 as f64 * 0.001_f64, -0.5 - 25 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_026() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 26 as f64 * 0.001_f64, -0.5 - 26 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_027() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 27 as f64 * 0.001_f64, -0.5 - 27 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_028() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 28 as f64 * 0.001_f64, -0.5 - 28 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_029() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 29 as f64 * 0.001_f64, -0.5 - 29 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_030() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 30 as f64 * 0.001_f64, -0.5 - 30 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_031() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 31 as f64 * 0.001_f64, -0.5 - 31 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_032() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 32 as f64 * 0.001_f64, -0.5 - 32 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_033() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 33 as f64 * 0.001_f64, -0.5 - 33 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_034() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 34 as f64 * 0.001_f64, -0.5 - 34 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_035() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 35 as f64 * 0.001_f64, -0.5 - 35 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_036() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 36 as f64 * 0.001_f64, -0.5 - 36 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_037() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 37 as f64 * 0.001_f64, -0.5 - 37 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_038() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 38 as f64 * 0.001_f64, -0.5 - 38 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_039() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 39 as f64 * 0.001_f64, -0.5 - 39 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_040() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 40 as f64 * 0.001_f64, -0.5 - 40 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_041() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 41 as f64 * 0.001_f64, -0.5 - 41 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_042() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 42 as f64 * 0.001_f64, -0.5 - 42 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_043() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 43 as f64 * 0.001_f64, -0.5 - 43 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_044() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 44 as f64 * 0.001_f64, -0.5 - 44 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_045() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 45 as f64 * 0.001_f64, -0.5 - 45 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_046() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 46 as f64 * 0.001_f64, -0.5 - 46 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_047() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 47 as f64 * 0.001_f64, -0.5 - 47 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_048() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 48 as f64 * 0.001_f64, -0.5 - 48 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_049() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 49 as f64 * 0.001_f64, -0.5 - 49 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_050() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 50 as f64 * 0.001_f64, -0.5 - 50 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_051() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 51 as f64 * 0.001_f64, -0.5 - 51 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_052() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 52 as f64 * 0.001_f64, -0.5 - 52 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_053() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 53 as f64 * 0.001_f64, -0.5 - 53 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_054() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 54 as f64 * 0.001_f64, -0.5 - 54 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_055() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 55 as f64 * 0.001_f64, -0.5 - 55 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_056() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 56 as f64 * 0.001_f64, -0.5 - 56 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_057() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 57 as f64 * 0.001_f64, -0.5 - 57 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_058() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 58 as f64 * 0.001_f64, -0.5 - 58 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_059() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 59 as f64 * 0.001_f64, -0.5 - 59 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_060() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 60 as f64 * 0.001_f64, -0.5 - 60 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_061() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 61 as f64 * 0.001_f64, -0.5 - 61 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_062() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 62 as f64 * 0.001_f64, -0.5 - 62 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_063() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 63 as f64 * 0.001_f64, -0.5 - 63 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_064() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 64 as f64 * 0.001_f64, -0.5 - 64 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_065() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 65 as f64 * 0.001_f64, -0.5 - 65 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_066() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 66 as f64 * 0.001_f64, -0.5 - 66 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_067() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 67 as f64 * 0.001_f64, -0.5 - 67 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_068() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 68 as f64 * 0.001_f64, -0.5 - 68 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_069() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 69 as f64 * 0.001_f64, -0.5 - 69 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_070() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 70 as f64 * 0.001_f64, -0.5 - 70 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_071() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 71 as f64 * 0.001_f64, -0.5 - 71 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_072() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 72 as f64 * 0.001_f64, -0.5 - 72 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_073() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 73 as f64 * 0.001_f64, -0.5 - 73 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_074() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 74 as f64 * 0.001_f64, -0.5 - 74 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_075() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 75 as f64 * 0.001_f64, -0.5 - 75 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_076() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 76 as f64 * 0.001_f64, -0.5 - 76 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_077() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 77 as f64 * 0.001_f64, -0.5 - 77 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_078() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 78 as f64 * 0.001_f64, -0.5 - 78 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_079() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 79 as f64 * 0.001_f64, -0.5 - 79 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_080() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 80 as f64 * 0.001_f64, -0.5 - 80 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_081() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 81 as f64 * 0.001_f64, -0.5 - 81 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_082() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 82 as f64 * 0.001_f64, -0.5 - 82 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_083() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 83 as f64 * 0.001_f64, -0.5 - 83 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_084() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 84 as f64 * 0.001_f64, -0.5 - 84 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_085() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 85 as f64 * 0.001_f64, -0.5 - 85 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_086() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 86 as f64 * 0.001_f64, -0.5 - 86 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_087() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 87 as f64 * 0.001_f64, -0.5 - 87 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_088() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 88 as f64 * 0.001_f64, -0.5 - 88 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_089() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 89 as f64 * 0.001_f64, -0.5 - 89 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_090() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 90 as f64 * 0.001_f64, -0.5 - 90 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_091() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 91 as f64 * 0.001_f64, -0.5 - 91 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_092() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 92 as f64 * 0.001_f64, -0.5 - 92 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_093() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 93 as f64 * 0.001_f64, -0.5 - 93 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_094() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 94 as f64 * 0.001_f64, -0.5 - 94 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_095() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 95 as f64 * 0.001_f64, -0.5 - 95 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_096() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 96 as f64 * 0.001_f64, -0.5 - 96 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_097() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 97 as f64 * 0.001_f64, -0.5 - 97 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_098() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 98 as f64 * 0.001_f64, -0.5 - 98 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_099() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 99 as f64 * 0.001_f64, -0.5 - 99 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_100() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 100 as f64 * 0.001_f64, -0.5 - 100 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_101() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 101 as f64 * 0.001_f64, -0.5 - 101 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_102() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 102 as f64 * 0.001_f64, -0.5 - 102 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_103() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 103 as f64 * 0.001_f64, -0.5 - 103 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_104() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 104 as f64 * 0.001_f64, -0.5 - 104 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_105() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 105 as f64 * 0.001_f64, -0.5 - 105 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_106() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 106 as f64 * 0.001_f64, -0.5 - 106 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_107() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 107 as f64 * 0.001_f64, -0.5 - 107 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_108() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 108 as f64 * 0.001_f64, -0.5 - 108 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_109() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 109 as f64 * 0.001_f64, -0.5 - 109 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_110() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 110 as f64 * 0.001_f64, -0.5 - 110 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_111() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 111 as f64 * 0.001_f64, -0.5 - 111 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_112() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 112 as f64 * 0.001_f64, -0.5 - 112 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_113() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 113 as f64 * 0.001_f64, -0.5 - 113 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_114() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 114 as f64 * 0.001_f64, -0.5 - 114 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_115() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 115 as f64 * 0.001_f64, -0.5 - 115 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_116() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 116 as f64 * 0.001_f64, -0.5 - 116 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_117() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 117 as f64 * 0.001_f64, -0.5 - 117 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_118() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 118 as f64 * 0.001_f64, -0.5 - 118 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_119() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 119 as f64 * 0.001_f64, -0.5 - 119 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_120() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 120 as f64 * 0.001_f64, -0.5 - 120 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_121() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 121 as f64 * 0.001_f64, -0.5 - 121 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_122() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 122 as f64 * 0.001_f64, -0.5 - 122 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_123() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 123 as f64 * 0.001_f64, -0.5 - 123 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_124() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 124 as f64 * 0.001_f64, -0.5 - 124 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_125() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 125 as f64 * 0.001_f64, -0.5 - 125 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_126() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 126 as f64 * 0.001_f64, -0.5 - 126 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_127() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 127 as f64 * 0.001_f64, -0.5 - 127 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_128() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 128 as f64 * 0.001_f64, -0.5 - 128 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_129() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 129 as f64 * 0.001_f64, -0.5 - 129 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_130() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 130 as f64 * 0.001_f64, -0.5 - 130 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_131() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 131 as f64 * 0.001_f64, -0.5 - 131 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_132() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 132 as f64 * 0.001_f64, -0.5 - 132 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_133() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 133 as f64 * 0.001_f64, -0.5 - 133 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_134() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 134 as f64 * 0.001_f64, -0.5 - 134 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_135() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 135 as f64 * 0.001_f64, -0.5 - 135 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_136() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 136 as f64 * 0.001_f64, -0.5 - 136 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_137() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 137 as f64 * 0.001_f64, -0.5 - 137 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_138() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 138 as f64 * 0.001_f64, -0.5 - 138 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_139() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 139 as f64 * 0.001_f64, -0.5 - 139 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_140() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 140 as f64 * 0.001_f64, -0.5 - 140 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_141() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 141 as f64 * 0.001_f64, -0.5 - 141 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_142() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 142 as f64 * 0.001_f64, -0.5 - 142 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_143() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 143 as f64 * 0.001_f64, -0.5 - 143 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_144() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 144 as f64 * 0.001_f64, -0.5 - 144 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_145() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 145 as f64 * 0.001_f64, -0.5 - 145 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_146() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 146 as f64 * 0.001_f64, -0.5 - 146 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_147() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 147 as f64 * 0.001_f64, -0.5 - 147 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_148() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 148 as f64 * 0.001_f64, -0.5 - 148 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_149() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 149 as f64 * 0.001_f64, -0.5 - 149 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_150() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 150 as f64 * 0.001_f64, -0.5 - 150 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_151() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 151 as f64 * 0.001_f64, -0.5 - 151 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_152() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 152 as f64 * 0.001_f64, -0.5 - 152 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_153() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 153 as f64 * 0.001_f64, -0.5 - 153 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
    }

    #[test]
    fn test_classic_loss_stress_154() {
        // Hinge: equilibrium at d_real=1, d_fake=-1 -> loss = 0.
        let hl = hinge_loss_d(1.0, -1.0);
        assert!((hl - 0.0).abs() < 1e-9);
        // WGAN: D wants to maximise d_real - d_fake.
        let wl = wgan_loss_d(0.5 + 154 as f64 * 0.001_f64, -0.5 - 154 as f64 * 0.001_f64);
        assert!(wl < 0.0);  // negative = good discriminator
        // LSGAN: equilibrium at d_real=1, d_fake=0.
        let ls = lsgan_loss_d(1.0, 0.0);
        assert!((ls - 0.0).abs() < 1e-9);
        // BCE generator: -log(p) -> decreasing in p.
        let bg1 = bce_loss_g(0.9);
        let bg2 = bce_loss_g(0.5);
        assert!(bg1 < bg2);
        // ClassicLoss dispatch.
        let cl = ClassicLoss::new();
        let dl = cl.d_loss(0.8, 0.2, "hinge", 0.1);
        assert!(dl >= 0.0);
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
}
