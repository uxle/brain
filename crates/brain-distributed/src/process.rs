//! # Worker Process Spawning & Simulation
//!
//! Helpers for spawning multi-rank simulations on a single host.

/// Runs a closure with simulated rank environment.
pub fn run_simulated_rank<F>(rank: usize, world_size: usize, f: F)
where
    F: FnOnce(usize, usize),
{
    f(rank, world_size);
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
