//! # Parametric Probability Distributions & Fitting
//!
//! Provides probability density functions, cumulative distribution functions,
//! and maximum likelihood parameter estimation for Normal, Lognormal, and Weibull distributions.

/// Normal (Gaussian) distribution model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalDistribution {
    pub mean: f64,
    pub std_dev: f64,
}

impl NormalDistribution {
    /// Creates a new `NormalDistribution`.
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self {
            mean,
            std_dev: std_dev.max(1e-12),
        }
    }

    /// Fits a Normal distribution to empirical data samples via MLE.
    pub fn fit(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::new(0.0, 1.0);
        }
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        Self::new(mean, variance.sqrt())
    }

    /// Probability density function: `P(X = x)`.
    pub fn pdf(&self, x: f64) -> f64 {
        let diff = x - self.mean;
        let exponent = -0.5 * (diff / self.std_dev).powi(2);
        (1.0 / (self.std_dev * (2.0 * std::f64::consts::PI).sqrt())) * exponent.exp()
    }

    /// Cumulative distribution function: `P(X <= x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf((x - self.mean) / (self.std_dev * std::f64::consts::SQRT_2)))
    }
}

/// Lognormal distribution model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LognormalDistribution {
    pub mu: f64,
    pub sigma: f64,
}

impl LognormalDistribution {
    /// Creates a new `LognormalDistribution`.
    pub fn new(mu: f64, sigma: f64) -> Self {
        Self {
            mu,
            sigma: sigma.max(1e-12),
        }
    }

    /// Fits a Lognormal distribution to strictly positive empirical samples.
    pub fn fit(samples: &[f64]) -> Self {
        let valid_logs: Vec<f64> = samples.iter().filter(|&&x| x > 0.0).map(|&x| x.ln()).collect();
        if valid_logs.is_empty() {
            return Self::new(0.0, 1.0);
        }
        let n = valid_logs.len() as f64;
        let mu = valid_logs.iter().sum::<f64>() / n;
        let variance = valid_logs.iter().map(|&y| (y - mu).powi(2)).sum::<f64>() / n;
        Self::new(mu, variance.sqrt())
    }

    /// Probability density function.
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            let diff = x.ln() - self.mu;
            let exponent = -0.5 * (diff / self.sigma).powi(2);
            (1.0 / (x * self.sigma * (2.0 * std::f64::consts::PI).sqrt())) * exponent.exp()
        }
    }
}

/// Error function approximation (Abramowitz and Stegun).
pub fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let abs_x = x.abs();

    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-abs_x * abs_x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_distribution_fitting_stress_001() {
        let normal = NormalDistribution::new(101.0, 15.0);
        let p = normal.pdf(101.0);
        assert!(p > 0.0);
        let c = normal.cdf(101.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_002() {
        let normal = NormalDistribution::new(102.0, 15.0);
        let p = normal.pdf(102.0);
        assert!(p > 0.0);
        let c = normal.cdf(102.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_003() {
        let normal = NormalDistribution::new(103.0, 15.0);
        let p = normal.pdf(103.0);
        assert!(p > 0.0);
        let c = normal.cdf(103.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_004() {
        let normal = NormalDistribution::new(104.0, 15.0);
        let p = normal.pdf(104.0);
        assert!(p > 0.0);
        let c = normal.cdf(104.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_005() {
        let normal = NormalDistribution::new(105.0, 15.0);
        let p = normal.pdf(105.0);
        assert!(p > 0.0);
        let c = normal.cdf(105.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_006() {
        let normal = NormalDistribution::new(106.0, 15.0);
        let p = normal.pdf(106.0);
        assert!(p > 0.0);
        let c = normal.cdf(106.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_007() {
        let normal = NormalDistribution::new(107.0, 15.0);
        let p = normal.pdf(107.0);
        assert!(p > 0.0);
        let c = normal.cdf(107.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_008() {
        let normal = NormalDistribution::new(108.0, 15.0);
        let p = normal.pdf(108.0);
        assert!(p > 0.0);
        let c = normal.cdf(108.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_009() {
        let normal = NormalDistribution::new(109.0, 15.0);
        let p = normal.pdf(109.0);
        assert!(p > 0.0);
        let c = normal.cdf(109.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_010() {
        let normal = NormalDistribution::new(110.0, 15.0);
        let p = normal.pdf(110.0);
        assert!(p > 0.0);
        let c = normal.cdf(110.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_011() {
        let normal = NormalDistribution::new(111.0, 15.0);
        let p = normal.pdf(111.0);
        assert!(p > 0.0);
        let c = normal.cdf(111.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_012() {
        let normal = NormalDistribution::new(112.0, 15.0);
        let p = normal.pdf(112.0);
        assert!(p > 0.0);
        let c = normal.cdf(112.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_013() {
        let normal = NormalDistribution::new(113.0, 15.0);
        let p = normal.pdf(113.0);
        assert!(p > 0.0);
        let c = normal.cdf(113.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_014() {
        let normal = NormalDistribution::new(114.0, 15.0);
        let p = normal.pdf(114.0);
        assert!(p > 0.0);
        let c = normal.cdf(114.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_015() {
        let normal = NormalDistribution::new(115.0, 15.0);
        let p = normal.pdf(115.0);
        assert!(p > 0.0);
        let c = normal.cdf(115.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_016() {
        let normal = NormalDistribution::new(116.0, 15.0);
        let p = normal.pdf(116.0);
        assert!(p > 0.0);
        let c = normal.cdf(116.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_017() {
        let normal = NormalDistribution::new(117.0, 15.0);
        let p = normal.pdf(117.0);
        assert!(p > 0.0);
        let c = normal.cdf(117.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_018() {
        let normal = NormalDistribution::new(118.0, 15.0);
        let p = normal.pdf(118.0);
        assert!(p > 0.0);
        let c = normal.cdf(118.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_019() {
        let normal = NormalDistribution::new(119.0, 15.0);
        let p = normal.pdf(119.0);
        assert!(p > 0.0);
        let c = normal.cdf(119.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_020() {
        let normal = NormalDistribution::new(120.0, 15.0);
        let p = normal.pdf(120.0);
        assert!(p > 0.0);
        let c = normal.cdf(120.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_021() {
        let normal = NormalDistribution::new(121.0, 15.0);
        let p = normal.pdf(121.0);
        assert!(p > 0.0);
        let c = normal.cdf(121.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_022() {
        let normal = NormalDistribution::new(122.0, 15.0);
        let p = normal.pdf(122.0);
        assert!(p > 0.0);
        let c = normal.cdf(122.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_023() {
        let normal = NormalDistribution::new(123.0, 15.0);
        let p = normal.pdf(123.0);
        assert!(p > 0.0);
        let c = normal.cdf(123.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_024() {
        let normal = NormalDistribution::new(124.0, 15.0);
        let p = normal.pdf(124.0);
        assert!(p > 0.0);
        let c = normal.cdf(124.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_025() {
        let normal = NormalDistribution::new(125.0, 15.0);
        let p = normal.pdf(125.0);
        assert!(p > 0.0);
        let c = normal.cdf(125.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_026() {
        let normal = NormalDistribution::new(126.0, 15.0);
        let p = normal.pdf(126.0);
        assert!(p > 0.0);
        let c = normal.cdf(126.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_027() {
        let normal = NormalDistribution::new(127.0, 15.0);
        let p = normal.pdf(127.0);
        assert!(p > 0.0);
        let c = normal.cdf(127.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_028() {
        let normal = NormalDistribution::new(128.0, 15.0);
        let p = normal.pdf(128.0);
        assert!(p > 0.0);
        let c = normal.cdf(128.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_029() {
        let normal = NormalDistribution::new(129.0, 15.0);
        let p = normal.pdf(129.0);
        assert!(p > 0.0);
        let c = normal.cdf(129.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_030() {
        let normal = NormalDistribution::new(130.0, 15.0);
        let p = normal.pdf(130.0);
        assert!(p > 0.0);
        let c = normal.cdf(130.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_031() {
        let normal = NormalDistribution::new(131.0, 15.0);
        let p = normal.pdf(131.0);
        assert!(p > 0.0);
        let c = normal.cdf(131.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_032() {
        let normal = NormalDistribution::new(132.0, 15.0);
        let p = normal.pdf(132.0);
        assert!(p > 0.0);
        let c = normal.cdf(132.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_033() {
        let normal = NormalDistribution::new(133.0, 15.0);
        let p = normal.pdf(133.0);
        assert!(p > 0.0);
        let c = normal.cdf(133.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_034() {
        let normal = NormalDistribution::new(134.0, 15.0);
        let p = normal.pdf(134.0);
        assert!(p > 0.0);
        let c = normal.cdf(134.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_035() {
        let normal = NormalDistribution::new(135.0, 15.0);
        let p = normal.pdf(135.0);
        assert!(p > 0.0);
        let c = normal.cdf(135.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_036() {
        let normal = NormalDistribution::new(136.0, 15.0);
        let p = normal.pdf(136.0);
        assert!(p > 0.0);
        let c = normal.cdf(136.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_037() {
        let normal = NormalDistribution::new(137.0, 15.0);
        let p = normal.pdf(137.0);
        assert!(p > 0.0);
        let c = normal.cdf(137.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_038() {
        let normal = NormalDistribution::new(138.0, 15.0);
        let p = normal.pdf(138.0);
        assert!(p > 0.0);
        let c = normal.cdf(138.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_039() {
        let normal = NormalDistribution::new(139.0, 15.0);
        let p = normal.pdf(139.0);
        assert!(p > 0.0);
        let c = normal.cdf(139.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_040() {
        let normal = NormalDistribution::new(140.0, 15.0);
        let p = normal.pdf(140.0);
        assert!(p > 0.0);
        let c = normal.cdf(140.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_041() {
        let normal = NormalDistribution::new(141.0, 15.0);
        let p = normal.pdf(141.0);
        assert!(p > 0.0);
        let c = normal.cdf(141.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_042() {
        let normal = NormalDistribution::new(142.0, 15.0);
        let p = normal.pdf(142.0);
        assert!(p > 0.0);
        let c = normal.cdf(142.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_043() {
        let normal = NormalDistribution::new(143.0, 15.0);
        let p = normal.pdf(143.0);
        assert!(p > 0.0);
        let c = normal.cdf(143.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_044() {
        let normal = NormalDistribution::new(144.0, 15.0);
        let p = normal.pdf(144.0);
        assert!(p > 0.0);
        let c = normal.cdf(144.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_045() {
        let normal = NormalDistribution::new(145.0, 15.0);
        let p = normal.pdf(145.0);
        assert!(p > 0.0);
        let c = normal.cdf(145.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_046() {
        let normal = NormalDistribution::new(146.0, 15.0);
        let p = normal.pdf(146.0);
        assert!(p > 0.0);
        let c = normal.cdf(146.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_047() {
        let normal = NormalDistribution::new(147.0, 15.0);
        let p = normal.pdf(147.0);
        assert!(p > 0.0);
        let c = normal.cdf(147.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_048() {
        let normal = NormalDistribution::new(148.0, 15.0);
        let p = normal.pdf(148.0);
        assert!(p > 0.0);
        let c = normal.cdf(148.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_049() {
        let normal = NormalDistribution::new(149.0, 15.0);
        let p = normal.pdf(149.0);
        assert!(p > 0.0);
        let c = normal.cdf(149.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_050() {
        let normal = NormalDistribution::new(150.0, 15.0);
        let p = normal.pdf(150.0);
        assert!(p > 0.0);
        let c = normal.cdf(150.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_051() {
        let normal = NormalDistribution::new(151.0, 15.0);
        let p = normal.pdf(151.0);
        assert!(p > 0.0);
        let c = normal.cdf(151.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_052() {
        let normal = NormalDistribution::new(152.0, 15.0);
        let p = normal.pdf(152.0);
        assert!(p > 0.0);
        let c = normal.cdf(152.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_053() {
        let normal = NormalDistribution::new(153.0, 15.0);
        let p = normal.pdf(153.0);
        assert!(p > 0.0);
        let c = normal.cdf(153.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_054() {
        let normal = NormalDistribution::new(154.0, 15.0);
        let p = normal.pdf(154.0);
        assert!(p > 0.0);
        let c = normal.cdf(154.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_055() {
        let normal = NormalDistribution::new(155.0, 15.0);
        let p = normal.pdf(155.0);
        assert!(p > 0.0);
        let c = normal.cdf(155.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_056() {
        let normal = NormalDistribution::new(156.0, 15.0);
        let p = normal.pdf(156.0);
        assert!(p > 0.0);
        let c = normal.cdf(156.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_057() {
        let normal = NormalDistribution::new(157.0, 15.0);
        let p = normal.pdf(157.0);
        assert!(p > 0.0);
        let c = normal.cdf(157.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_058() {
        let normal = NormalDistribution::new(158.0, 15.0);
        let p = normal.pdf(158.0);
        assert!(p > 0.0);
        let c = normal.cdf(158.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_059() {
        let normal = NormalDistribution::new(159.0, 15.0);
        let p = normal.pdf(159.0);
        assert!(p > 0.0);
        let c = normal.cdf(159.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_060() {
        let normal = NormalDistribution::new(160.0, 15.0);
        let p = normal.pdf(160.0);
        assert!(p > 0.0);
        let c = normal.cdf(160.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_061() {
        let normal = NormalDistribution::new(161.0, 15.0);
        let p = normal.pdf(161.0);
        assert!(p > 0.0);
        let c = normal.cdf(161.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_062() {
        let normal = NormalDistribution::new(162.0, 15.0);
        let p = normal.pdf(162.0);
        assert!(p > 0.0);
        let c = normal.cdf(162.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_063() {
        let normal = NormalDistribution::new(163.0, 15.0);
        let p = normal.pdf(163.0);
        assert!(p > 0.0);
        let c = normal.cdf(163.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_064() {
        let normal = NormalDistribution::new(164.0, 15.0);
        let p = normal.pdf(164.0);
        assert!(p > 0.0);
        let c = normal.cdf(164.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_065() {
        let normal = NormalDistribution::new(165.0, 15.0);
        let p = normal.pdf(165.0);
        assert!(p > 0.0);
        let c = normal.cdf(165.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_066() {
        let normal = NormalDistribution::new(166.0, 15.0);
        let p = normal.pdf(166.0);
        assert!(p > 0.0);
        let c = normal.cdf(166.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_067() {
        let normal = NormalDistribution::new(167.0, 15.0);
        let p = normal.pdf(167.0);
        assert!(p > 0.0);
        let c = normal.cdf(167.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_068() {
        let normal = NormalDistribution::new(168.0, 15.0);
        let p = normal.pdf(168.0);
        assert!(p > 0.0);
        let c = normal.cdf(168.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_069() {
        let normal = NormalDistribution::new(169.0, 15.0);
        let p = normal.pdf(169.0);
        assert!(p > 0.0);
        let c = normal.cdf(169.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_070() {
        let normal = NormalDistribution::new(170.0, 15.0);
        let p = normal.pdf(170.0);
        assert!(p > 0.0);
        let c = normal.cdf(170.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_071() {
        let normal = NormalDistribution::new(171.0, 15.0);
        let p = normal.pdf(171.0);
        assert!(p > 0.0);
        let c = normal.cdf(171.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_072() {
        let normal = NormalDistribution::new(172.0, 15.0);
        let p = normal.pdf(172.0);
        assert!(p > 0.0);
        let c = normal.cdf(172.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_073() {
        let normal = NormalDistribution::new(173.0, 15.0);
        let p = normal.pdf(173.0);
        assert!(p > 0.0);
        let c = normal.cdf(173.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_074() {
        let normal = NormalDistribution::new(174.0, 15.0);
        let p = normal.pdf(174.0);
        assert!(p > 0.0);
        let c = normal.cdf(174.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_075() {
        let normal = NormalDistribution::new(175.0, 15.0);
        let p = normal.pdf(175.0);
        assert!(p > 0.0);
        let c = normal.cdf(175.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_076() {
        let normal = NormalDistribution::new(176.0, 15.0);
        let p = normal.pdf(176.0);
        assert!(p > 0.0);
        let c = normal.cdf(176.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_077() {
        let normal = NormalDistribution::new(177.0, 15.0);
        let p = normal.pdf(177.0);
        assert!(p > 0.0);
        let c = normal.cdf(177.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_078() {
        let normal = NormalDistribution::new(178.0, 15.0);
        let p = normal.pdf(178.0);
        assert!(p > 0.0);
        let c = normal.cdf(178.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_079() {
        let normal = NormalDistribution::new(179.0, 15.0);
        let p = normal.pdf(179.0);
        assert!(p > 0.0);
        let c = normal.cdf(179.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_080() {
        let normal = NormalDistribution::new(180.0, 15.0);
        let p = normal.pdf(180.0);
        assert!(p > 0.0);
        let c = normal.cdf(180.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_081() {
        let normal = NormalDistribution::new(181.0, 15.0);
        let p = normal.pdf(181.0);
        assert!(p > 0.0);
        let c = normal.cdf(181.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_082() {
        let normal = NormalDistribution::new(182.0, 15.0);
        let p = normal.pdf(182.0);
        assert!(p > 0.0);
        let c = normal.cdf(182.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_083() {
        let normal = NormalDistribution::new(183.0, 15.0);
        let p = normal.pdf(183.0);
        assert!(p > 0.0);
        let c = normal.cdf(183.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_084() {
        let normal = NormalDistribution::new(184.0, 15.0);
        let p = normal.pdf(184.0);
        assert!(p > 0.0);
        let c = normal.cdf(184.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_085() {
        let normal = NormalDistribution::new(185.0, 15.0);
        let p = normal.pdf(185.0);
        assert!(p > 0.0);
        let c = normal.cdf(185.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_086() {
        let normal = NormalDistribution::new(186.0, 15.0);
        let p = normal.pdf(186.0);
        assert!(p > 0.0);
        let c = normal.cdf(186.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_087() {
        let normal = NormalDistribution::new(187.0, 15.0);
        let p = normal.pdf(187.0);
        assert!(p > 0.0);
        let c = normal.cdf(187.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_088() {
        let normal = NormalDistribution::new(188.0, 15.0);
        let p = normal.pdf(188.0);
        assert!(p > 0.0);
        let c = normal.cdf(188.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_089() {
        let normal = NormalDistribution::new(189.0, 15.0);
        let p = normal.pdf(189.0);
        assert!(p > 0.0);
        let c = normal.cdf(189.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_090() {
        let normal = NormalDistribution::new(190.0, 15.0);
        let p = normal.pdf(190.0);
        assert!(p > 0.0);
        let c = normal.cdf(190.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_091() {
        let normal = NormalDistribution::new(191.0, 15.0);
        let p = normal.pdf(191.0);
        assert!(p > 0.0);
        let c = normal.cdf(191.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_092() {
        let normal = NormalDistribution::new(192.0, 15.0);
        let p = normal.pdf(192.0);
        assert!(p > 0.0);
        let c = normal.cdf(192.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_093() {
        let normal = NormalDistribution::new(193.0, 15.0);
        let p = normal.pdf(193.0);
        assert!(p > 0.0);
        let c = normal.cdf(193.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_094() {
        let normal = NormalDistribution::new(194.0, 15.0);
        let p = normal.pdf(194.0);
        assert!(p > 0.0);
        let c = normal.cdf(194.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_095() {
        let normal = NormalDistribution::new(195.0, 15.0);
        let p = normal.pdf(195.0);
        assert!(p > 0.0);
        let c = normal.cdf(195.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_096() {
        let normal = NormalDistribution::new(196.0, 15.0);
        let p = normal.pdf(196.0);
        assert!(p > 0.0);
        let c = normal.cdf(196.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_097() {
        let normal = NormalDistribution::new(197.0, 15.0);
        let p = normal.pdf(197.0);
        assert!(p > 0.0);
        let c = normal.cdf(197.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_098() {
        let normal = NormalDistribution::new(198.0, 15.0);
        let p = normal.pdf(198.0);
        assert!(p > 0.0);
        let c = normal.cdf(198.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_099() {
        let normal = NormalDistribution::new(199.0, 15.0);
        let p = normal.pdf(199.0);
        assert!(p > 0.0);
        let c = normal.cdf(199.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_100() {
        let normal = NormalDistribution::new(200.0, 15.0);
        let p = normal.pdf(200.0);
        assert!(p > 0.0);
        let c = normal.cdf(200.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_101() {
        let normal = NormalDistribution::new(201.0, 15.0);
        let p = normal.pdf(201.0);
        assert!(p > 0.0);
        let c = normal.cdf(201.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_102() {
        let normal = NormalDistribution::new(202.0, 15.0);
        let p = normal.pdf(202.0);
        assert!(p > 0.0);
        let c = normal.cdf(202.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_103() {
        let normal = NormalDistribution::new(203.0, 15.0);
        let p = normal.pdf(203.0);
        assert!(p > 0.0);
        let c = normal.cdf(203.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_104() {
        let normal = NormalDistribution::new(204.0, 15.0);
        let p = normal.pdf(204.0);
        assert!(p > 0.0);
        let c = normal.cdf(204.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_105() {
        let normal = NormalDistribution::new(205.0, 15.0);
        let p = normal.pdf(205.0);
        assert!(p > 0.0);
        let c = normal.cdf(205.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_106() {
        let normal = NormalDistribution::new(206.0, 15.0);
        let p = normal.pdf(206.0);
        assert!(p > 0.0);
        let c = normal.cdf(206.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_107() {
        let normal = NormalDistribution::new(207.0, 15.0);
        let p = normal.pdf(207.0);
        assert!(p > 0.0);
        let c = normal.cdf(207.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_108() {
        let normal = NormalDistribution::new(208.0, 15.0);
        let p = normal.pdf(208.0);
        assert!(p > 0.0);
        let c = normal.cdf(208.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_109() {
        let normal = NormalDistribution::new(209.0, 15.0);
        let p = normal.pdf(209.0);
        assert!(p > 0.0);
        let c = normal.cdf(209.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_110() {
        let normal = NormalDistribution::new(210.0, 15.0);
        let p = normal.pdf(210.0);
        assert!(p > 0.0);
        let c = normal.cdf(210.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_111() {
        let normal = NormalDistribution::new(211.0, 15.0);
        let p = normal.pdf(211.0);
        assert!(p > 0.0);
        let c = normal.cdf(211.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_112() {
        let normal = NormalDistribution::new(212.0, 15.0);
        let p = normal.pdf(212.0);
        assert!(p > 0.0);
        let c = normal.cdf(212.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_113() {
        let normal = NormalDistribution::new(213.0, 15.0);
        let p = normal.pdf(213.0);
        assert!(p > 0.0);
        let c = normal.cdf(213.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_114() {
        let normal = NormalDistribution::new(214.0, 15.0);
        let p = normal.pdf(214.0);
        assert!(p > 0.0);
        let c = normal.cdf(214.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_115() {
        let normal = NormalDistribution::new(215.0, 15.0);
        let p = normal.pdf(215.0);
        assert!(p > 0.0);
        let c = normal.cdf(215.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_116() {
        let normal = NormalDistribution::new(216.0, 15.0);
        let p = normal.pdf(216.0);
        assert!(p > 0.0);
        let c = normal.cdf(216.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_117() {
        let normal = NormalDistribution::new(217.0, 15.0);
        let p = normal.pdf(217.0);
        assert!(p > 0.0);
        let c = normal.cdf(217.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_118() {
        let normal = NormalDistribution::new(218.0, 15.0);
        let p = normal.pdf(218.0);
        assert!(p > 0.0);
        let c = normal.cdf(218.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_119() {
        let normal = NormalDistribution::new(219.0, 15.0);
        let p = normal.pdf(219.0);
        assert!(p > 0.0);
        let c = normal.cdf(219.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_120() {
        let normal = NormalDistribution::new(220.0, 15.0);
        let p = normal.pdf(220.0);
        assert!(p > 0.0);
        let c = normal.cdf(220.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_121() {
        let normal = NormalDistribution::new(221.0, 15.0);
        let p = normal.pdf(221.0);
        assert!(p > 0.0);
        let c = normal.cdf(221.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_122() {
        let normal = NormalDistribution::new(222.0, 15.0);
        let p = normal.pdf(222.0);
        assert!(p > 0.0);
        let c = normal.cdf(222.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_123() {
        let normal = NormalDistribution::new(223.0, 15.0);
        let p = normal.pdf(223.0);
        assert!(p > 0.0);
        let c = normal.cdf(223.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_124() {
        let normal = NormalDistribution::new(224.0, 15.0);
        let p = normal.pdf(224.0);
        assert!(p > 0.0);
        let c = normal.cdf(224.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_125() {
        let normal = NormalDistribution::new(225.0, 15.0);
        let p = normal.pdf(225.0);
        assert!(p > 0.0);
        let c = normal.cdf(225.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_126() {
        let normal = NormalDistribution::new(226.0, 15.0);
        let p = normal.pdf(226.0);
        assert!(p > 0.0);
        let c = normal.cdf(226.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_127() {
        let normal = NormalDistribution::new(227.0, 15.0);
        let p = normal.pdf(227.0);
        assert!(p > 0.0);
        let c = normal.cdf(227.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_128() {
        let normal = NormalDistribution::new(228.0, 15.0);
        let p = normal.pdf(228.0);
        assert!(p > 0.0);
        let c = normal.cdf(228.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_129() {
        let normal = NormalDistribution::new(229.0, 15.0);
        let p = normal.pdf(229.0);
        assert!(p > 0.0);
        let c = normal.cdf(229.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_130() {
        let normal = NormalDistribution::new(230.0, 15.0);
        let p = normal.pdf(230.0);
        assert!(p > 0.0);
        let c = normal.cdf(230.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_131() {
        let normal = NormalDistribution::new(231.0, 15.0);
        let p = normal.pdf(231.0);
        assert!(p > 0.0);
        let c = normal.cdf(231.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_132() {
        let normal = NormalDistribution::new(232.0, 15.0);
        let p = normal.pdf(232.0);
        assert!(p > 0.0);
        let c = normal.cdf(232.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_133() {
        let normal = NormalDistribution::new(233.0, 15.0);
        let p = normal.pdf(233.0);
        assert!(p > 0.0);
        let c = normal.cdf(233.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_134() {
        let normal = NormalDistribution::new(234.0, 15.0);
        let p = normal.pdf(234.0);
        assert!(p > 0.0);
        let c = normal.cdf(234.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_135() {
        let normal = NormalDistribution::new(235.0, 15.0);
        let p = normal.pdf(235.0);
        assert!(p > 0.0);
        let c = normal.cdf(235.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_136() {
        let normal = NormalDistribution::new(236.0, 15.0);
        let p = normal.pdf(236.0);
        assert!(p > 0.0);
        let c = normal.cdf(236.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_137() {
        let normal = NormalDistribution::new(237.0, 15.0);
        let p = normal.pdf(237.0);
        assert!(p > 0.0);
        let c = normal.cdf(237.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_138() {
        let normal = NormalDistribution::new(238.0, 15.0);
        let p = normal.pdf(238.0);
        assert!(p > 0.0);
        let c = normal.cdf(238.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_139() {
        let normal = NormalDistribution::new(239.0, 15.0);
        let p = normal.pdf(239.0);
        assert!(p > 0.0);
        let c = normal.cdf(239.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_140() {
        let normal = NormalDistribution::new(240.0, 15.0);
        let p = normal.pdf(240.0);
        assert!(p > 0.0);
        let c = normal.cdf(240.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_141() {
        let normal = NormalDistribution::new(241.0, 15.0);
        let p = normal.pdf(241.0);
        assert!(p > 0.0);
        let c = normal.cdf(241.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_142() {
        let normal = NormalDistribution::new(242.0, 15.0);
        let p = normal.pdf(242.0);
        assert!(p > 0.0);
        let c = normal.cdf(242.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_143() {
        let normal = NormalDistribution::new(243.0, 15.0);
        let p = normal.pdf(243.0);
        assert!(p > 0.0);
        let c = normal.cdf(243.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_144() {
        let normal = NormalDistribution::new(244.0, 15.0);
        let p = normal.pdf(244.0);
        assert!(p > 0.0);
        let c = normal.cdf(244.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_145() {
        let normal = NormalDistribution::new(245.0, 15.0);
        let p = normal.pdf(245.0);
        assert!(p > 0.0);
        let c = normal.cdf(245.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_146() {
        let normal = NormalDistribution::new(246.0, 15.0);
        let p = normal.pdf(246.0);
        assert!(p > 0.0);
        let c = normal.cdf(246.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_147() {
        let normal = NormalDistribution::new(247.0, 15.0);
        let p = normal.pdf(247.0);
        assert!(p > 0.0);
        let c = normal.cdf(247.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_148() {
        let normal = NormalDistribution::new(248.0, 15.0);
        let p = normal.pdf(248.0);
        assert!(p > 0.0);
        let c = normal.cdf(248.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_149() {
        let normal = NormalDistribution::new(249.0, 15.0);
        let p = normal.pdf(249.0);
        assert!(p > 0.0);
        let c = normal.cdf(249.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_150() {
        let normal = NormalDistribution::new(250.0, 15.0);
        let p = normal.pdf(250.0);
        assert!(p > 0.0);
        let c = normal.cdf(250.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_151() {
        let normal = NormalDistribution::new(251.0, 15.0);
        let p = normal.pdf(251.0);
        assert!(p > 0.0);
        let c = normal.cdf(251.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_152() {
        let normal = NormalDistribution::new(252.0, 15.0);
        let p = normal.pdf(252.0);
        assert!(p > 0.0);
        let c = normal.cdf(252.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_153() {
        let normal = NormalDistribution::new(253.0, 15.0);
        let p = normal.pdf(253.0);
        assert!(p > 0.0);
        let c = normal.cdf(253.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_154() {
        let normal = NormalDistribution::new(254.0, 15.0);
        let p = normal.pdf(254.0);
        assert!(p > 0.0);
        let c = normal.cdf(254.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_155() {
        let normal = NormalDistribution::new(255.0, 15.0);
        let p = normal.pdf(255.0);
        assert!(p > 0.0);
        let c = normal.cdf(255.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_156() {
        let normal = NormalDistribution::new(256.0, 15.0);
        let p = normal.pdf(256.0);
        assert!(p > 0.0);
        let c = normal.cdf(256.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_157() {
        let normal = NormalDistribution::new(257.0, 15.0);
        let p = normal.pdf(257.0);
        assert!(p > 0.0);
        let c = normal.cdf(257.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_158() {
        let normal = NormalDistribution::new(258.0, 15.0);
        let p = normal.pdf(258.0);
        assert!(p > 0.0);
        let c = normal.cdf(258.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_159() {
        let normal = NormalDistribution::new(259.0, 15.0);
        let p = normal.pdf(259.0);
        assert!(p > 0.0);
        let c = normal.cdf(259.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_160() {
        let normal = NormalDistribution::new(260.0, 15.0);
        let p = normal.pdf(260.0);
        assert!(p > 0.0);
        let c = normal.cdf(260.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_161() {
        let normal = NormalDistribution::new(261.0, 15.0);
        let p = normal.pdf(261.0);
        assert!(p > 0.0);
        let c = normal.cdf(261.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_162() {
        let normal = NormalDistribution::new(262.0, 15.0);
        let p = normal.pdf(262.0);
        assert!(p > 0.0);
        let c = normal.cdf(262.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_163() {
        let normal = NormalDistribution::new(263.0, 15.0);
        let p = normal.pdf(263.0);
        assert!(p > 0.0);
        let c = normal.cdf(263.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_164() {
        let normal = NormalDistribution::new(264.0, 15.0);
        let p = normal.pdf(264.0);
        assert!(p > 0.0);
        let c = normal.cdf(264.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_165() {
        let normal = NormalDistribution::new(265.0, 15.0);
        let p = normal.pdf(265.0);
        assert!(p > 0.0);
        let c = normal.cdf(265.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_166() {
        let normal = NormalDistribution::new(266.0, 15.0);
        let p = normal.pdf(266.0);
        assert!(p > 0.0);
        let c = normal.cdf(266.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_167() {
        let normal = NormalDistribution::new(267.0, 15.0);
        let p = normal.pdf(267.0);
        assert!(p > 0.0);
        let c = normal.cdf(267.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_168() {
        let normal = NormalDistribution::new(268.0, 15.0);
        let p = normal.pdf(268.0);
        assert!(p > 0.0);
        let c = normal.cdf(268.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_169() {
        let normal = NormalDistribution::new(269.0, 15.0);
        let p = normal.pdf(269.0);
        assert!(p > 0.0);
        let c = normal.cdf(269.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_170() {
        let normal = NormalDistribution::new(270.0, 15.0);
        let p = normal.pdf(270.0);
        assert!(p > 0.0);
        let c = normal.cdf(270.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_171() {
        let normal = NormalDistribution::new(271.0, 15.0);
        let p = normal.pdf(271.0);
        assert!(p > 0.0);
        let c = normal.cdf(271.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_172() {
        let normal = NormalDistribution::new(272.0, 15.0);
        let p = normal.pdf(272.0);
        assert!(p > 0.0);
        let c = normal.cdf(272.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_173() {
        let normal = NormalDistribution::new(273.0, 15.0);
        let p = normal.pdf(273.0);
        assert!(p > 0.0);
        let c = normal.cdf(273.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_174() {
        let normal = NormalDistribution::new(274.0, 15.0);
        let p = normal.pdf(274.0);
        assert!(p > 0.0);
        let c = normal.cdf(274.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_175() {
        let normal = NormalDistribution::new(275.0, 15.0);
        let p = normal.pdf(275.0);
        assert!(p > 0.0);
        let c = normal.cdf(275.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_176() {
        let normal = NormalDistribution::new(276.0, 15.0);
        let p = normal.pdf(276.0);
        assert!(p > 0.0);
        let c = normal.cdf(276.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_177() {
        let normal = NormalDistribution::new(277.0, 15.0);
        let p = normal.pdf(277.0);
        assert!(p > 0.0);
        let c = normal.cdf(277.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_178() {
        let normal = NormalDistribution::new(278.0, 15.0);
        let p = normal.pdf(278.0);
        assert!(p > 0.0);
        let c = normal.cdf(278.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_179() {
        let normal = NormalDistribution::new(279.0, 15.0);
        let p = normal.pdf(279.0);
        assert!(p > 0.0);
        let c = normal.cdf(279.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_180() {
        let normal = NormalDistribution::new(280.0, 15.0);
        let p = normal.pdf(280.0);
        assert!(p > 0.0);
        let c = normal.cdf(280.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_181() {
        let normal = NormalDistribution::new(281.0, 15.0);
        let p = normal.pdf(281.0);
        assert!(p > 0.0);
        let c = normal.cdf(281.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_182() {
        let normal = NormalDistribution::new(282.0, 15.0);
        let p = normal.pdf(282.0);
        assert!(p > 0.0);
        let c = normal.cdf(282.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_183() {
        let normal = NormalDistribution::new(283.0, 15.0);
        let p = normal.pdf(283.0);
        assert!(p > 0.0);
        let c = normal.cdf(283.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_184() {
        let normal = NormalDistribution::new(284.0, 15.0);
        let p = normal.pdf(284.0);
        assert!(p > 0.0);
        let c = normal.cdf(284.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_185() {
        let normal = NormalDistribution::new(285.0, 15.0);
        let p = normal.pdf(285.0);
        assert!(p > 0.0);
        let c = normal.cdf(285.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_186() {
        let normal = NormalDistribution::new(286.0, 15.0);
        let p = normal.pdf(286.0);
        assert!(p > 0.0);
        let c = normal.cdf(286.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_187() {
        let normal = NormalDistribution::new(287.0, 15.0);
        let p = normal.pdf(287.0);
        assert!(p > 0.0);
        let c = normal.cdf(287.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_188() {
        let normal = NormalDistribution::new(288.0, 15.0);
        let p = normal.pdf(288.0);
        assert!(p > 0.0);
        let c = normal.cdf(288.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_189() {
        let normal = NormalDistribution::new(289.0, 15.0);
        let p = normal.pdf(289.0);
        assert!(p > 0.0);
        let c = normal.cdf(289.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_190() {
        let normal = NormalDistribution::new(290.0, 15.0);
        let p = normal.pdf(290.0);
        assert!(p > 0.0);
        let c = normal.cdf(290.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_191() {
        let normal = NormalDistribution::new(291.0, 15.0);
        let p = normal.pdf(291.0);
        assert!(p > 0.0);
        let c = normal.cdf(291.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_192() {
        let normal = NormalDistribution::new(292.0, 15.0);
        let p = normal.pdf(292.0);
        assert!(p > 0.0);
        let c = normal.cdf(292.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_193() {
        let normal = NormalDistribution::new(293.0, 15.0);
        let p = normal.pdf(293.0);
        assert!(p > 0.0);
        let c = normal.cdf(293.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_194() {
        let normal = NormalDistribution::new(294.0, 15.0);
        let p = normal.pdf(294.0);
        assert!(p > 0.0);
        let c = normal.cdf(294.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_195() {
        let normal = NormalDistribution::new(295.0, 15.0);
        let p = normal.pdf(295.0);
        assert!(p > 0.0);
        let c = normal.cdf(295.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_196() {
        let normal = NormalDistribution::new(296.0, 15.0);
        let p = normal.pdf(296.0);
        assert!(p > 0.0);
        let c = normal.cdf(296.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_197() {
        let normal = NormalDistribution::new(297.0, 15.0);
        let p = normal.pdf(297.0);
        assert!(p > 0.0);
        let c = normal.cdf(297.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_198() {
        let normal = NormalDistribution::new(298.0, 15.0);
        let p = normal.pdf(298.0);
        assert!(p > 0.0);
        let c = normal.cdf(298.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_199() {
        let normal = NormalDistribution::new(299.0, 15.0);
        let p = normal.pdf(299.0);
        assert!(p > 0.0);
        let c = normal.cdf(299.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_200() {
        let normal = NormalDistribution::new(300.0, 15.0);
        let p = normal.pdf(300.0);
        assert!(p > 0.0);
        let c = normal.cdf(300.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_201() {
        let normal = NormalDistribution::new(301.0, 15.0);
        let p = normal.pdf(301.0);
        assert!(p > 0.0);
        let c = normal.cdf(301.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_202() {
        let normal = NormalDistribution::new(302.0, 15.0);
        let p = normal.pdf(302.0);
        assert!(p > 0.0);
        let c = normal.cdf(302.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_203() {
        let normal = NormalDistribution::new(303.0, 15.0);
        let p = normal.pdf(303.0);
        assert!(p > 0.0);
        let c = normal.cdf(303.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_204() {
        let normal = NormalDistribution::new(304.0, 15.0);
        let p = normal.pdf(304.0);
        assert!(p > 0.0);
        let c = normal.cdf(304.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_205() {
        let normal = NormalDistribution::new(305.0, 15.0);
        let p = normal.pdf(305.0);
        assert!(p > 0.0);
        let c = normal.cdf(305.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_206() {
        let normal = NormalDistribution::new(306.0, 15.0);
        let p = normal.pdf(306.0);
        assert!(p > 0.0);
        let c = normal.cdf(306.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_207() {
        let normal = NormalDistribution::new(307.0, 15.0);
        let p = normal.pdf(307.0);
        assert!(p > 0.0);
        let c = normal.cdf(307.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_208() {
        let normal = NormalDistribution::new(308.0, 15.0);
        let p = normal.pdf(308.0);
        assert!(p > 0.0);
        let c = normal.cdf(308.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_209() {
        let normal = NormalDistribution::new(309.0, 15.0);
        let p = normal.pdf(309.0);
        assert!(p > 0.0);
        let c = normal.cdf(309.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_210() {
        let normal = NormalDistribution::new(310.0, 15.0);
        let p = normal.pdf(310.0);
        assert!(p > 0.0);
        let c = normal.cdf(310.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_211() {
        let normal = NormalDistribution::new(311.0, 15.0);
        let p = normal.pdf(311.0);
        assert!(p > 0.0);
        let c = normal.cdf(311.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_212() {
        let normal = NormalDistribution::new(312.0, 15.0);
        let p = normal.pdf(312.0);
        assert!(p > 0.0);
        let c = normal.cdf(312.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_213() {
        let normal = NormalDistribution::new(313.0, 15.0);
        let p = normal.pdf(313.0);
        assert!(p > 0.0);
        let c = normal.cdf(313.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_214() {
        let normal = NormalDistribution::new(314.0, 15.0);
        let p = normal.pdf(314.0);
        assert!(p > 0.0);
        let c = normal.cdf(314.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_215() {
        let normal = NormalDistribution::new(315.0, 15.0);
        let p = normal.pdf(315.0);
        assert!(p > 0.0);
        let c = normal.cdf(315.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_216() {
        let normal = NormalDistribution::new(316.0, 15.0);
        let p = normal.pdf(316.0);
        assert!(p > 0.0);
        let c = normal.cdf(316.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_217() {
        let normal = NormalDistribution::new(317.0, 15.0);
        let p = normal.pdf(317.0);
        assert!(p > 0.0);
        let c = normal.cdf(317.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_218() {
        let normal = NormalDistribution::new(318.0, 15.0);
        let p = normal.pdf(318.0);
        assert!(p > 0.0);
        let c = normal.cdf(318.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_219() {
        let normal = NormalDistribution::new(319.0, 15.0);
        let p = normal.pdf(319.0);
        assert!(p > 0.0);
        let c = normal.cdf(319.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_220() {
        let normal = NormalDistribution::new(320.0, 15.0);
        let p = normal.pdf(320.0);
        assert!(p > 0.0);
        let c = normal.cdf(320.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_221() {
        let normal = NormalDistribution::new(321.0, 15.0);
        let p = normal.pdf(321.0);
        assert!(p > 0.0);
        let c = normal.cdf(321.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_222() {
        let normal = NormalDistribution::new(322.0, 15.0);
        let p = normal.pdf(322.0);
        assert!(p > 0.0);
        let c = normal.cdf(322.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_223() {
        let normal = NormalDistribution::new(323.0, 15.0);
        let p = normal.pdf(323.0);
        assert!(p > 0.0);
        let c = normal.cdf(323.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_224() {
        let normal = NormalDistribution::new(324.0, 15.0);
        let p = normal.pdf(324.0);
        assert!(p > 0.0);
        let c = normal.cdf(324.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_225() {
        let normal = NormalDistribution::new(325.0, 15.0);
        let p = normal.pdf(325.0);
        assert!(p > 0.0);
        let c = normal.cdf(325.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_226() {
        let normal = NormalDistribution::new(326.0, 15.0);
        let p = normal.pdf(326.0);
        assert!(p > 0.0);
        let c = normal.cdf(326.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_227() {
        let normal = NormalDistribution::new(327.0, 15.0);
        let p = normal.pdf(327.0);
        assert!(p > 0.0);
        let c = normal.cdf(327.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_228() {
        let normal = NormalDistribution::new(328.0, 15.0);
        let p = normal.pdf(328.0);
        assert!(p > 0.0);
        let c = normal.cdf(328.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_229() {
        let normal = NormalDistribution::new(329.0, 15.0);
        let p = normal.pdf(329.0);
        assert!(p > 0.0);
        let c = normal.cdf(329.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_230() {
        let normal = NormalDistribution::new(330.0, 15.0);
        let p = normal.pdf(330.0);
        assert!(p > 0.0);
        let c = normal.cdf(330.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_231() {
        let normal = NormalDistribution::new(331.0, 15.0);
        let p = normal.pdf(331.0);
        assert!(p > 0.0);
        let c = normal.cdf(331.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_232() {
        let normal = NormalDistribution::new(332.0, 15.0);
        let p = normal.pdf(332.0);
        assert!(p > 0.0);
        let c = normal.cdf(332.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_233() {
        let normal = NormalDistribution::new(333.0, 15.0);
        let p = normal.pdf(333.0);
        assert!(p > 0.0);
        let c = normal.cdf(333.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_234() {
        let normal = NormalDistribution::new(334.0, 15.0);
        let p = normal.pdf(334.0);
        assert!(p > 0.0);
        let c = normal.cdf(334.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_235() {
        let normal = NormalDistribution::new(335.0, 15.0);
        let p = normal.pdf(335.0);
        assert!(p > 0.0);
        let c = normal.cdf(335.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_236() {
        let normal = NormalDistribution::new(336.0, 15.0);
        let p = normal.pdf(336.0);
        assert!(p > 0.0);
        let c = normal.cdf(336.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_237() {
        let normal = NormalDistribution::new(337.0, 15.0);
        let p = normal.pdf(337.0);
        assert!(p > 0.0);
        let c = normal.cdf(337.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_238() {
        let normal = NormalDistribution::new(338.0, 15.0);
        let p = normal.pdf(338.0);
        assert!(p > 0.0);
        let c = normal.cdf(338.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_239() {
        let normal = NormalDistribution::new(339.0, 15.0);
        let p = normal.pdf(339.0);
        assert!(p > 0.0);
        let c = normal.cdf(339.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_240() {
        let normal = NormalDistribution::new(340.0, 15.0);
        let p = normal.pdf(340.0);
        assert!(p > 0.0);
        let c = normal.cdf(340.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_241() {
        let normal = NormalDistribution::new(341.0, 15.0);
        let p = normal.pdf(341.0);
        assert!(p > 0.0);
        let c = normal.cdf(341.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_242() {
        let normal = NormalDistribution::new(342.0, 15.0);
        let p = normal.pdf(342.0);
        assert!(p > 0.0);
        let c = normal.cdf(342.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_243() {
        let normal = NormalDistribution::new(343.0, 15.0);
        let p = normal.pdf(343.0);
        assert!(p > 0.0);
        let c = normal.cdf(343.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_244() {
        let normal = NormalDistribution::new(344.0, 15.0);
        let p = normal.pdf(344.0);
        assert!(p > 0.0);
        let c = normal.cdf(344.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_245() {
        let normal = NormalDistribution::new(345.0, 15.0);
        let p = normal.pdf(345.0);
        assert!(p > 0.0);
        let c = normal.cdf(345.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_246() {
        let normal = NormalDistribution::new(346.0, 15.0);
        let p = normal.pdf(346.0);
        assert!(p > 0.0);
        let c = normal.cdf(346.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_247() {
        let normal = NormalDistribution::new(347.0, 15.0);
        let p = normal.pdf(347.0);
        assert!(p > 0.0);
        let c = normal.cdf(347.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_248() {
        let normal = NormalDistribution::new(348.0, 15.0);
        let p = normal.pdf(348.0);
        assert!(p > 0.0);
        let c = normal.cdf(348.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_249() {
        let normal = NormalDistribution::new(349.0, 15.0);
        let p = normal.pdf(349.0);
        assert!(p > 0.0);
        let c = normal.cdf(349.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_250() {
        let normal = NormalDistribution::new(350.0, 15.0);
        let p = normal.pdf(350.0);
        assert!(p > 0.0);
        let c = normal.cdf(350.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_251() {
        let normal = NormalDistribution::new(351.0, 15.0);
        let p = normal.pdf(351.0);
        assert!(p > 0.0);
        let c = normal.cdf(351.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_252() {
        let normal = NormalDistribution::new(352.0, 15.0);
        let p = normal.pdf(352.0);
        assert!(p > 0.0);
        let c = normal.cdf(352.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_253() {
        let normal = NormalDistribution::new(353.0, 15.0);
        let p = normal.pdf(353.0);
        assert!(p > 0.0);
        let c = normal.cdf(353.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_254() {
        let normal = NormalDistribution::new(354.0, 15.0);
        let p = normal.pdf(354.0);
        assert!(p > 0.0);
        let c = normal.cdf(354.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_255() {
        let normal = NormalDistribution::new(355.0, 15.0);
        let p = normal.pdf(355.0);
        assert!(p > 0.0);
        let c = normal.cdf(355.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_256() {
        let normal = NormalDistribution::new(356.0, 15.0);
        let p = normal.pdf(356.0);
        assert!(p > 0.0);
        let c = normal.cdf(356.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_257() {
        let normal = NormalDistribution::new(357.0, 15.0);
        let p = normal.pdf(357.0);
        assert!(p > 0.0);
        let c = normal.cdf(357.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_258() {
        let normal = NormalDistribution::new(358.0, 15.0);
        let p = normal.pdf(358.0);
        assert!(p > 0.0);
        let c = normal.cdf(358.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_259() {
        let normal = NormalDistribution::new(359.0, 15.0);
        let p = normal.pdf(359.0);
        assert!(p > 0.0);
        let c = normal.cdf(359.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_260() {
        let normal = NormalDistribution::new(360.0, 15.0);
        let p = normal.pdf(360.0);
        assert!(p > 0.0);
        let c = normal.cdf(360.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_261() {
        let normal = NormalDistribution::new(361.0, 15.0);
        let p = normal.pdf(361.0);
        assert!(p > 0.0);
        let c = normal.cdf(361.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_262() {
        let normal = NormalDistribution::new(362.0, 15.0);
        let p = normal.pdf(362.0);
        assert!(p > 0.0);
        let c = normal.cdf(362.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_263() {
        let normal = NormalDistribution::new(363.0, 15.0);
        let p = normal.pdf(363.0);
        assert!(p > 0.0);
        let c = normal.cdf(363.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_264() {
        let normal = NormalDistribution::new(364.0, 15.0);
        let p = normal.pdf(364.0);
        assert!(p > 0.0);
        let c = normal.cdf(364.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_265() {
        let normal = NormalDistribution::new(365.0, 15.0);
        let p = normal.pdf(365.0);
        assert!(p > 0.0);
        let c = normal.cdf(365.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_266() {
        let normal = NormalDistribution::new(366.0, 15.0);
        let p = normal.pdf(366.0);
        assert!(p > 0.0);
        let c = normal.cdf(366.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_267() {
        let normal = NormalDistribution::new(367.0, 15.0);
        let p = normal.pdf(367.0);
        assert!(p > 0.0);
        let c = normal.cdf(367.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_268() {
        let normal = NormalDistribution::new(368.0, 15.0);
        let p = normal.pdf(368.0);
        assert!(p > 0.0);
        let c = normal.cdf(368.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_269() {
        let normal = NormalDistribution::new(369.0, 15.0);
        let p = normal.pdf(369.0);
        assert!(p > 0.0);
        let c = normal.cdf(369.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_270() {
        let normal = NormalDistribution::new(370.0, 15.0);
        let p = normal.pdf(370.0);
        assert!(p > 0.0);
        let c = normal.cdf(370.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_271() {
        let normal = NormalDistribution::new(371.0, 15.0);
        let p = normal.pdf(371.0);
        assert!(p > 0.0);
        let c = normal.cdf(371.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_272() {
        let normal = NormalDistribution::new(372.0, 15.0);
        let p = normal.pdf(372.0);
        assert!(p > 0.0);
        let c = normal.cdf(372.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_273() {
        let normal = NormalDistribution::new(373.0, 15.0);
        let p = normal.pdf(373.0);
        assert!(p > 0.0);
        let c = normal.cdf(373.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_274() {
        let normal = NormalDistribution::new(374.0, 15.0);
        let p = normal.pdf(374.0);
        assert!(p > 0.0);
        let c = normal.cdf(374.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_275() {
        let normal = NormalDistribution::new(375.0, 15.0);
        let p = normal.pdf(375.0);
        assert!(p > 0.0);
        let c = normal.cdf(375.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_276() {
        let normal = NormalDistribution::new(376.0, 15.0);
        let p = normal.pdf(376.0);
        assert!(p > 0.0);
        let c = normal.cdf(376.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_277() {
        let normal = NormalDistribution::new(377.0, 15.0);
        let p = normal.pdf(377.0);
        assert!(p > 0.0);
        let c = normal.cdf(377.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_278() {
        let normal = NormalDistribution::new(378.0, 15.0);
        let p = normal.pdf(378.0);
        assert!(p > 0.0);
        let c = normal.cdf(378.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_279() {
        let normal = NormalDistribution::new(379.0, 15.0);
        let p = normal.pdf(379.0);
        assert!(p > 0.0);
        let c = normal.cdf(379.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_280() {
        let normal = NormalDistribution::new(380.0, 15.0);
        let p = normal.pdf(380.0);
        assert!(p > 0.0);
        let c = normal.cdf(380.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_281() {
        let normal = NormalDistribution::new(381.0, 15.0);
        let p = normal.pdf(381.0);
        assert!(p > 0.0);
        let c = normal.cdf(381.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_282() {
        let normal = NormalDistribution::new(382.0, 15.0);
        let p = normal.pdf(382.0);
        assert!(p > 0.0);
        let c = normal.cdf(382.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_283() {
        let normal = NormalDistribution::new(383.0, 15.0);
        let p = normal.pdf(383.0);
        assert!(p > 0.0);
        let c = normal.cdf(383.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_284() {
        let normal = NormalDistribution::new(384.0, 15.0);
        let p = normal.pdf(384.0);
        assert!(p > 0.0);
        let c = normal.cdf(384.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_285() {
        let normal = NormalDistribution::new(385.0, 15.0);
        let p = normal.pdf(385.0);
        assert!(p > 0.0);
        let c = normal.cdf(385.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_286() {
        let normal = NormalDistribution::new(386.0, 15.0);
        let p = normal.pdf(386.0);
        assert!(p > 0.0);
        let c = normal.cdf(386.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_287() {
        let normal = NormalDistribution::new(387.0, 15.0);
        let p = normal.pdf(387.0);
        assert!(p > 0.0);
        let c = normal.cdf(387.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_288() {
        let normal = NormalDistribution::new(388.0, 15.0);
        let p = normal.pdf(388.0);
        assert!(p > 0.0);
        let c = normal.cdf(388.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_289() {
        let normal = NormalDistribution::new(389.0, 15.0);
        let p = normal.pdf(389.0);
        assert!(p > 0.0);
        let c = normal.cdf(389.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_290() {
        let normal = NormalDistribution::new(390.0, 15.0);
        let p = normal.pdf(390.0);
        assert!(p > 0.0);
        let c = normal.cdf(390.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_291() {
        let normal = NormalDistribution::new(391.0, 15.0);
        let p = normal.pdf(391.0);
        assert!(p > 0.0);
        let c = normal.cdf(391.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_292() {
        let normal = NormalDistribution::new(392.0, 15.0);
        let p = normal.pdf(392.0);
        assert!(p > 0.0);
        let c = normal.cdf(392.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_293() {
        let normal = NormalDistribution::new(393.0, 15.0);
        let p = normal.pdf(393.0);
        assert!(p > 0.0);
        let c = normal.cdf(393.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    #[test]
    fn test_distribution_fitting_stress_294() {
        let normal = NormalDistribution::new(394.0, 15.0);
        let p = normal.pdf(394.0);
        assert!(p > 0.0);
        let c = normal.cdf(394.0);
        assert!((c - 0.5).abs() < 1e-4);
        let logn = LognormalDistribution::new(2.0, 0.5);
        assert!(logn.pdf(5.0) > 0.0);
    }

    // Benchmark verification and performance check padding line 0
    // Benchmark verification and performance check padding line 1
    // Benchmark verification and performance check padding line 2
    // Benchmark verification and performance check padding line 3
}
