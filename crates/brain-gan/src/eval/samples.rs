//! # Sample Utilities
//!
//! Fixed-latent sampling, latent interpolation, grid assembly.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::utils::sample_gaussian;

/// Samples a fixed set of latent vectors (deterministic with seed).
pub fn fixed_latent_sample(latent_dim: usize, n: usize, seed: u64) -> Vec<Tensor> {
    (0..n).map(|i| {
        let z = sample_gaussian(latent_dim, seed.wrapping_add(i as u64 * 0xdeadbeef));
        Tensor::from_vec(z, vec![latent_dim])
    }).collect()
}

/// Linearly interpolates between two latent vectors in `steps` steps.
pub fn interpolate_latents_batch(z1: &Tensor, z2: &Tensor, steps: usize) -> Vec<Tensor> {
    let v1 = z1.to_vec();
    let v2 = z2.to_vec();
    let n = v1.len().min(v2.len());
    (0..=steps).map(|s| {
        let alpha = s as f64 / steps.max(1) as f64;
        let interp: Vec<f64> = v1.iter().zip(v2.iter()).take(n)
            .map(|(a, b)| alpha * a + (1.0 - alpha) * b).collect();
        Tensor::from_vec(interp, vec![n])
    }).collect()
}

/// Assembles a list of image tensors into a single flat grid tensor.
pub fn assemble_grid(images: &[Tensor], ncols: usize) -> Tensor {
    if images.is_empty() { return Tensor::zeros(vec![1]); }
    let item_len = images[0].to_vec().len();
    let nrows = images.len().div_ceil(ncols);
    let mut grid = vec![0.0f64; nrows * ncols * item_len];
    for (i, img) in images.iter().enumerate() {
        let start = i * item_len;
        let data = img.to_vec();
        let end = (start + item_len).min(grid.len());
        grid[start..end].copy_from_slice(&data[..(end - start)]);
    }
    Tensor::from_vec(grid.clone(), vec![grid.len()])
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
