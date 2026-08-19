//! # Top-Level Neuroevolution Runner
//!
//! Convenient evolutionary loop runner: `run_evolution`, `evolve_generation`, `best_genome`.
#![allow(missing_docs)]

use crate::core::{EvoConfig, EvoResult, EvoError};
use crate::fitness::FitnessFn;
use crate::ga::Ga;

/// High-level runner executing an evolutionary optimization process to completion.
pub fn run_evolution<F: FitnessFn>(
    config: &EvoConfig,
    fitness_fn: &F,
    seed: u64,
) -> EvoResult<(Vec<f64>, f64)> {
    config.validate().map_err(EvoError::InvalidConfig)?;

    let mut ga = Ga::new(config.clone(), seed);
    let result = ga.run(fitness_fn)?;
    Ok((result.best_genome, result.best_fitness))
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
