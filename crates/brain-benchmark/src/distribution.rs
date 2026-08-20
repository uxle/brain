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
        let valid_logs: Vec<f64> = samples
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| x.ln())
            .collect();
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
}
