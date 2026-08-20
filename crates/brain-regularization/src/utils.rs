//! # Statistical & RNG Utilities
//!
//! Fast XorShift64 PRNG, running statistics accumulators (Welford's algorithm), and reduction helpers.
#![allow(
    missing_docs,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown
    )]
    use super::*;
    use crate::augment::*;
    use crate::config::*;
    use crate::consistency::*;
    use crate::core::*;
    use crate::curriculum::*;
    use crate::decay::*;
    use crate::dropout::*;
    use crate::dropout_uncertainty::*;
    use crate::earlystop::*;
    use crate::label_smooth::*;
    use crate::normalization::*;
    use crate::ops::*;
    use crate::perturb::*;
    use crate::r#impl::*;
    use crate::registry::*;
    use crate::regularizers::*;
    use crate::rules::*;
    use crate::stopping::*;
    use crate::train_hooks::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
