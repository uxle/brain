//! # Memory-Safe Dataset Shuffling
//!
//! Provides permutation index shuffling, windowed shuffling, and deterministic seeded generators.

/// Generates a pseudo-random permutation of indices `0..len` using a deterministic seed.
pub fn shuffle_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    let mut rng_state = seed.wrapping_add(0x9e3779b97f4a7c15);

    for i in (1..len).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state % (i as u64 + 1)) as usize;
        indices.swap(i, j);
    }

    indices
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
