//! # Genome Representation & Initialization
//!
//! Continuous parameter vector, fitness container, and random initialization strategies.
#![allow(missing_docs)]

pub mod encoding;
pub use encoding::{EncodingKind, GenomeEncoding};

use crate::utils::FastRng;

/// Represents an individual evolutionary genome.
#[derive(Debug, Clone, PartialEq)]
pub struct Genome {
    pub genes: Vec<f64>,
    pub fitness: Option<f64>,
    pub generation: usize,
}

impl Genome {
    pub fn new(genes: Vec<f64>) -> Self {
        Self {
            genes,
            fitness: None,
            generation: 0,
        }
    }

    pub fn random_uniform(dim: usize, min_val: f64, max_val: f64, rng: &mut FastRng) -> Self {
        let genes: Vec<f64> = (0..dim)
            .map(|_| rng.sample_range(min_val, max_val))
            .collect();
        Self::new(genes)
    }

    pub fn random_gaussian(dim: usize, mean: f64, std_dev: f64, rng: &mut FastRng) -> Self {
        let genes: Vec<f64> = (0..dim)
            .map(|_| rng.sample_gaussian(mean, std_dev))
            .collect();
        Self::new(genes)
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
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
