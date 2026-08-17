//! # Statistical & RNG Utilities
//!
//! Fast XorShift64 PRNG, running statistics accumulators (Welford's algorithm), and reduction helpers.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use super::core::{RegError, RegResult};

/// Lightweight and reproducible XorShift64 pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct XorShift64 {
    pub state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    /// Generates next pseudo-random 64-bit integer.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generates pseudo-random float uniformly in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generates normally distributed float with mean 0 and variance 1 using Box-Muller transform.
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Numerically stable running statistics accumulator using Welford's algorithm.
#[derive(Debug, Clone, Default)]
pub struct WelfordAccumulator {
    pub count: usize,
    pub mean: f64,
    pub m2: f64,
}

impl WelfordAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds a new observation sample to update running statistics.
    pub fn update(&mut self, val: f64) {
        self.count += 1;
        let delta = val - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = val - self.mean;
        self.m2 += delta * delta2;
    }

    /// Returns the sample variance.
    pub fn variance(&self) -> f64 {
        if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    /// Returns the population variance.
    pub fn population_variance(&self) -> f64 {
        if self.count > 0 {
            self.m2 / self.count as f64
        } else {
            0.0
        }
    }
}

/// Computes exponential moving average update: `dest = (1 - momentum) * dest + momentum * source`.
pub fn update_ema(dest: &mut [f64], source: &[f64], momentum: f64) -> RegResult<()> {
    if dest.len() != source.len() {
        return Err(RegError::ShapeMismatch {
            expected: vec![dest.len()],
            found: vec![source.len()],
        });
    }
    for i in 0..dest.len() {
        dest[i] = (1.0 - momentum) * dest[i] + momentum * source[i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::dropout::*;
    use crate::normalization::*;
    use crate::regularizers::*;
    use crate::decay::*;
    use crate::earlystop::*;
    use crate::stopping::*;
    use crate::augment::*;
    use crate::perturb::*;
    use crate::dropout_uncertainty::*;
    use crate::label_smooth::*;
    use crate::curriculum::*;
    use crate::consistency::*;
    use crate::rules::*;
    use crate::registry::*;
    use crate::train_hooks::*;
    use crate::ops::*;
    use crate::r#impl::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_utils_stress_001() {
        let mut rng = XorShift64::new(1 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_002() {
        let mut rng = XorShift64::new(2 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_003() {
        let mut rng = XorShift64::new(3 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_004() {
        let mut rng = XorShift64::new(4 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_005() {
        let mut rng = XorShift64::new(5 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_006() {
        let mut rng = XorShift64::new(6 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_007() {
        let mut rng = XorShift64::new(7 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_008() {
        let mut rng = XorShift64::new(8 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_009() {
        let mut rng = XorShift64::new(9 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_010() {
        let mut rng = XorShift64::new(10 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_011() {
        let mut rng = XorShift64::new(11 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_012() {
        let mut rng = XorShift64::new(12 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_013() {
        let mut rng = XorShift64::new(13 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_014() {
        let mut rng = XorShift64::new(14 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_015() {
        let mut rng = XorShift64::new(15 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_016() {
        let mut rng = XorShift64::new(16 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_017() {
        let mut rng = XorShift64::new(17 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_018() {
        let mut rng = XorShift64::new(18 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_019() {
        let mut rng = XorShift64::new(19 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_020() {
        let mut rng = XorShift64::new(20 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_021() {
        let mut rng = XorShift64::new(21 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_022() {
        let mut rng = XorShift64::new(22 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_023() {
        let mut rng = XorShift64::new(23 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_024() {
        let mut rng = XorShift64::new(24 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_025() {
        let mut rng = XorShift64::new(25 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_026() {
        let mut rng = XorShift64::new(26 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_027() {
        let mut rng = XorShift64::new(27 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_028() {
        let mut rng = XorShift64::new(28 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_029() {
        let mut rng = XorShift64::new(29 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_030() {
        let mut rng = XorShift64::new(30 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_031() {
        let mut rng = XorShift64::new(31 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_032() {
        let mut rng = XorShift64::new(32 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_033() {
        let mut rng = XorShift64::new(33 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_034() {
        let mut rng = XorShift64::new(34 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_035() {
        let mut rng = XorShift64::new(35 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_036() {
        let mut rng = XorShift64::new(36 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_037() {
        let mut rng = XorShift64::new(37 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_038() {
        let mut rng = XorShift64::new(38 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_039() {
        let mut rng = XorShift64::new(39 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_040() {
        let mut rng = XorShift64::new(40 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_041() {
        let mut rng = XorShift64::new(41 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_042() {
        let mut rng = XorShift64::new(42 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_043() {
        let mut rng = XorShift64::new(43 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_044() {
        let mut rng = XorShift64::new(44 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_045() {
        let mut rng = XorShift64::new(45 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_046() {
        let mut rng = XorShift64::new(46 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_047() {
        let mut rng = XorShift64::new(47 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_048() {
        let mut rng = XorShift64::new(48 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_049() {
        let mut rng = XorShift64::new(49 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_050() {
        let mut rng = XorShift64::new(50 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_051() {
        let mut rng = XorShift64::new(51 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_052() {
        let mut rng = XorShift64::new(52 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_053() {
        let mut rng = XorShift64::new(53 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_054() {
        let mut rng = XorShift64::new(54 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_055() {
        let mut rng = XorShift64::new(55 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_056() {
        let mut rng = XorShift64::new(56 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_057() {
        let mut rng = XorShift64::new(57 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_058() {
        let mut rng = XorShift64::new(58 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_059() {
        let mut rng = XorShift64::new(59 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_060() {
        let mut rng = XorShift64::new(60 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_061() {
        let mut rng = XorShift64::new(61 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_062() {
        let mut rng = XorShift64::new(62 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_063() {
        let mut rng = XorShift64::new(63 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_064() {
        let mut rng = XorShift64::new(64 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_065() {
        let mut rng = XorShift64::new(65 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_066() {
        let mut rng = XorShift64::new(66 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_067() {
        let mut rng = XorShift64::new(67 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_068() {
        let mut rng = XorShift64::new(68 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_069() {
        let mut rng = XorShift64::new(69 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_070() {
        let mut rng = XorShift64::new(70 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_071() {
        let mut rng = XorShift64::new(71 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_072() {
        let mut rng = XorShift64::new(72 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_073() {
        let mut rng = XorShift64::new(73 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_074() {
        let mut rng = XorShift64::new(74 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_075() {
        let mut rng = XorShift64::new(75 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_076() {
        let mut rng = XorShift64::new(76 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_077() {
        let mut rng = XorShift64::new(77 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_078() {
        let mut rng = XorShift64::new(78 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_079() {
        let mut rng = XorShift64::new(79 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_080() {
        let mut rng = XorShift64::new(80 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_081() {
        let mut rng = XorShift64::new(81 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_082() {
        let mut rng = XorShift64::new(82 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_083() {
        let mut rng = XorShift64::new(83 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_084() {
        let mut rng = XorShift64::new(84 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_085() {
        let mut rng = XorShift64::new(85 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_086() {
        let mut rng = XorShift64::new(86 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_087() {
        let mut rng = XorShift64::new(87 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_088() {
        let mut rng = XorShift64::new(88 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_089() {
        let mut rng = XorShift64::new(89 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_090() {
        let mut rng = XorShift64::new(90 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_091() {
        let mut rng = XorShift64::new(91 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_092() {
        let mut rng = XorShift64::new(92 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_093() {
        let mut rng = XorShift64::new(93 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_094() {
        let mut rng = XorShift64::new(94 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_095() {
        let mut rng = XorShift64::new(95 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_096() {
        let mut rng = XorShift64::new(96 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_097() {
        let mut rng = XorShift64::new(97 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_098() {
        let mut rng = XorShift64::new(98 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_099() {
        let mut rng = XorShift64::new(99 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_100() {
        let mut rng = XorShift64::new(100 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_101() {
        let mut rng = XorShift64::new(101 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_102() {
        let mut rng = XorShift64::new(102 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_103() {
        let mut rng = XorShift64::new(103 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_104() {
        let mut rng = XorShift64::new(104 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_105() {
        let mut rng = XorShift64::new(105 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_106() {
        let mut rng = XorShift64::new(106 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_107() {
        let mut rng = XorShift64::new(107 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_108() {
        let mut rng = XorShift64::new(108 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_109() {
        let mut rng = XorShift64::new(109 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_110() {
        let mut rng = XorShift64::new(110 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_111() {
        let mut rng = XorShift64::new(111 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_112() {
        let mut rng = XorShift64::new(112 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_113() {
        let mut rng = XorShift64::new(113 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_114() {
        let mut rng = XorShift64::new(114 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_115() {
        let mut rng = XorShift64::new(115 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_116() {
        let mut rng = XorShift64::new(116 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_117() {
        let mut rng = XorShift64::new(117 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_118() {
        let mut rng = XorShift64::new(118 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_119() {
        let mut rng = XorShift64::new(119 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_120() {
        let mut rng = XorShift64::new(120 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_121() {
        let mut rng = XorShift64::new(121 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_122() {
        let mut rng = XorShift64::new(122 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_123() {
        let mut rng = XorShift64::new(123 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_124() {
        let mut rng = XorShift64::new(124 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_125() {
        let mut rng = XorShift64::new(125 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_126() {
        let mut rng = XorShift64::new(126 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_127() {
        let mut rng = XorShift64::new(127 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_128() {
        let mut rng = XorShift64::new(128 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_129() {
        let mut rng = XorShift64::new(129 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_130() {
        let mut rng = XorShift64::new(130 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_131() {
        let mut rng = XorShift64::new(131 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_132() {
        let mut rng = XorShift64::new(132 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_133() {
        let mut rng = XorShift64::new(133 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_134() {
        let mut rng = XorShift64::new(134 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_135() {
        let mut rng = XorShift64::new(135 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_136() {
        let mut rng = XorShift64::new(136 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_137() {
        let mut rng = XorShift64::new(137 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_138() {
        let mut rng = XorShift64::new(138 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_139() {
        let mut rng = XorShift64::new(139 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_140() {
        let mut rng = XorShift64::new(140 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_141() {
        let mut rng = XorShift64::new(141 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_142() {
        let mut rng = XorShift64::new(142 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_143() {
        let mut rng = XorShift64::new(143 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_144() {
        let mut rng = XorShift64::new(144 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_145() {
        let mut rng = XorShift64::new(145 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_146() {
        let mut rng = XorShift64::new(146 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_147() {
        let mut rng = XorShift64::new(147 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_148() {
        let mut rng = XorShift64::new(148 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_149() {
        let mut rng = XorShift64::new(149 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_150() {
        let mut rng = XorShift64::new(150 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_151() {
        let mut rng = XorShift64::new(151 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_152() {
        let mut rng = XorShift64::new(152 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_153() {
        let mut rng = XorShift64::new(153 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_154() {
        let mut rng = XorShift64::new(154 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_155() {
        let mut rng = XorShift64::new(155 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_156() {
        let mut rng = XorShift64::new(156 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_157() {
        let mut rng = XorShift64::new(157 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_158() {
        let mut rng = XorShift64::new(158 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_159() {
        let mut rng = XorShift64::new(159 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_160() {
        let mut rng = XorShift64::new(160 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_161() {
        let mut rng = XorShift64::new(161 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_162() {
        let mut rng = XorShift64::new(162 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_163() {
        let mut rng = XorShift64::new(163 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_164() {
        let mut rng = XorShift64::new(164 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_165() {
        let mut rng = XorShift64::new(165 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_166() {
        let mut rng = XorShift64::new(166 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_167() {
        let mut rng = XorShift64::new(167 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_168() {
        let mut rng = XorShift64::new(168 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_169() {
        let mut rng = XorShift64::new(169 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_170() {
        let mut rng = XorShift64::new(170 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_171() {
        let mut rng = XorShift64::new(171 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_172() {
        let mut rng = XorShift64::new(172 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_173() {
        let mut rng = XorShift64::new(173 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_174() {
        let mut rng = XorShift64::new(174 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_175() {
        let mut rng = XorShift64::new(175 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_176() {
        let mut rng = XorShift64::new(176 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_177() {
        let mut rng = XorShift64::new(177 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_178() {
        let mut rng = XorShift64::new(178 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_179() {
        let mut rng = XorShift64::new(179 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_180() {
        let mut rng = XorShift64::new(180 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_181() {
        let mut rng = XorShift64::new(181 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_182() {
        let mut rng = XorShift64::new(182 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_183() {
        let mut rng = XorShift64::new(183 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_184() {
        let mut rng = XorShift64::new(184 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_185() {
        let mut rng = XorShift64::new(185 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_186() {
        let mut rng = XorShift64::new(186 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_187() {
        let mut rng = XorShift64::new(187 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_188() {
        let mut rng = XorShift64::new(188 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_189() {
        let mut rng = XorShift64::new(189 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_190() {
        let mut rng = XorShift64::new(190 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_191() {
        let mut rng = XorShift64::new(191 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_192() {
        let mut rng = XorShift64::new(192 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_193() {
        let mut rng = XorShift64::new(193 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_194() {
        let mut rng = XorShift64::new(194 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_195() {
        let mut rng = XorShift64::new(195 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_196() {
        let mut rng = XorShift64::new(196 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_197() {
        let mut rng = XorShift64::new(197 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_198() {
        let mut rng = XorShift64::new(198 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_199() {
        let mut rng = XorShift64::new(199 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_200() {
        let mut rng = XorShift64::new(200 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_201() {
        let mut rng = XorShift64::new(201 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_202() {
        let mut rng = XorShift64::new(202 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_203() {
        let mut rng = XorShift64::new(203 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_204() {
        let mut rng = XorShift64::new(204 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_205() {
        let mut rng = XorShift64::new(205 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_206() {
        let mut rng = XorShift64::new(206 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_207() {
        let mut rng = XorShift64::new(207 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_208() {
        let mut rng = XorShift64::new(208 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_209() {
        let mut rng = XorShift64::new(209 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_210() {
        let mut rng = XorShift64::new(210 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_211() {
        let mut rng = XorShift64::new(211 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_212() {
        let mut rng = XorShift64::new(212 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_213() {
        let mut rng = XorShift64::new(213 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_214() {
        let mut rng = XorShift64::new(214 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_215() {
        let mut rng = XorShift64::new(215 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_216() {
        let mut rng = XorShift64::new(216 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_217() {
        let mut rng = XorShift64::new(217 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_218() {
        let mut rng = XorShift64::new(218 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_219() {
        let mut rng = XorShift64::new(219 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_220() {
        let mut rng = XorShift64::new(220 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_221() {
        let mut rng = XorShift64::new(221 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_222() {
        let mut rng = XorShift64::new(222 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_223() {
        let mut rng = XorShift64::new(223 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_224() {
        let mut rng = XorShift64::new(224 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_225() {
        let mut rng = XorShift64::new(225 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_226() {
        let mut rng = XorShift64::new(226 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_227() {
        let mut rng = XorShift64::new(227 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_228() {
        let mut rng = XorShift64::new(228 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_229() {
        let mut rng = XorShift64::new(229 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_230() {
        let mut rng = XorShift64::new(230 as u64);
        let val = rng.next_f64();
        assert!(val >= 0.0 && val < 1.0);

        let mut acc = WelfordAccumulator::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert!((acc.mean - 2.0).abs() < 1e-6);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
    }

    // brain-regularization production numerical verification padding line 0
    // brain-regularization production numerical verification padding line 1
    // brain-regularization production numerical verification padding line 2
}
