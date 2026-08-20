//! # Evolutionary Utilities & Pseudo-Random Helpers
//!
//! Deterministic XorShift RNG, index sampling, elitism sorting, and fitness ranking.
#![allow(missing_docs)]

/// Fast, deterministic, zero-dependency XorShift64 pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    pub fn seed(seed: u64) -> Self {
        let s = if seed == 0 { 0x853c49e6748fea9b } else { seed };
        Self { state: s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    #[inline]
    pub fn sample_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }

    #[inline]
    pub fn sample_gaussian(&mut self, mean: f64, std_dev: f64) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + z0 * std_dev
    }
}

/// Returns sorted indices of individuals by fitness descending (higher is better).
pub fn rank_fitness(fitnesses: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..fitnesses.len()).collect();
    indices.sort_by(|&a, &b| {
        fitnesses[b]
            .partial_cmp(&fitnesses[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
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
