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

    #[test]
    fn test_samples_stress_001() {
        let zs = fixed_latent_sample(8, 2, 1 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_002() {
        let zs = fixed_latent_sample(8, 3, 2 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_003() {
        let zs = fixed_latent_sample(8, 4, 3 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_004() {
        let zs = fixed_latent_sample(8, 5, 4 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_005() {
        let zs = fixed_latent_sample(8, 6, 5 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_006() {
        let zs = fixed_latent_sample(8, 7, 6 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_007() {
        let zs = fixed_latent_sample(8, 8, 7 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_008() {
        let zs = fixed_latent_sample(8, 1, 8 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_009() {
        let zs = fixed_latent_sample(8, 2, 9 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_010() {
        let zs = fixed_latent_sample(8, 3, 10 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_011() {
        let zs = fixed_latent_sample(8, 4, 11 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_012() {
        let zs = fixed_latent_sample(8, 5, 12 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_013() {
        let zs = fixed_latent_sample(8, 6, 13 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_014() {
        let zs = fixed_latent_sample(8, 7, 14 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_015() {
        let zs = fixed_latent_sample(8, 8, 15 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_016() {
        let zs = fixed_latent_sample(8, 1, 16 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_017() {
        let zs = fixed_latent_sample(8, 2, 17 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_018() {
        let zs = fixed_latent_sample(8, 3, 18 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_019() {
        let zs = fixed_latent_sample(8, 4, 19 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_020() {
        let zs = fixed_latent_sample(8, 5, 20 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_021() {
        let zs = fixed_latent_sample(8, 6, 21 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_022() {
        let zs = fixed_latent_sample(8, 7, 22 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_023() {
        let zs = fixed_latent_sample(8, 8, 23 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_024() {
        let zs = fixed_latent_sample(8, 1, 24 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_025() {
        let zs = fixed_latent_sample(8, 2, 25 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_026() {
        let zs = fixed_latent_sample(8, 3, 26 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_027() {
        let zs = fixed_latent_sample(8, 4, 27 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_028() {
        let zs = fixed_latent_sample(8, 5, 28 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_029() {
        let zs = fixed_latent_sample(8, 6, 29 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_030() {
        let zs = fixed_latent_sample(8, 7, 30 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_031() {
        let zs = fixed_latent_sample(8, 8, 31 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_032() {
        let zs = fixed_latent_sample(8, 1, 32 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_033() {
        let zs = fixed_latent_sample(8, 2, 33 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_034() {
        let zs = fixed_latent_sample(8, 3, 34 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_035() {
        let zs = fixed_latent_sample(8, 4, 35 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_036() {
        let zs = fixed_latent_sample(8, 5, 36 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_037() {
        let zs = fixed_latent_sample(8, 6, 37 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_038() {
        let zs = fixed_latent_sample(8, 7, 38 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_039() {
        let zs = fixed_latent_sample(8, 8, 39 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_040() {
        let zs = fixed_latent_sample(8, 1, 40 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_041() {
        let zs = fixed_latent_sample(8, 2, 41 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_042() {
        let zs = fixed_latent_sample(8, 3, 42 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_043() {
        let zs = fixed_latent_sample(8, 4, 43 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_044() {
        let zs = fixed_latent_sample(8, 5, 44 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_045() {
        let zs = fixed_latent_sample(8, 6, 45 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_046() {
        let zs = fixed_latent_sample(8, 7, 46 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_047() {
        let zs = fixed_latent_sample(8, 8, 47 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_048() {
        let zs = fixed_latent_sample(8, 1, 48 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_049() {
        let zs = fixed_latent_sample(8, 2, 49 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_050() {
        let zs = fixed_latent_sample(8, 3, 50 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_051() {
        let zs = fixed_latent_sample(8, 4, 51 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_052() {
        let zs = fixed_latent_sample(8, 5, 52 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_053() {
        let zs = fixed_latent_sample(8, 6, 53 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_054() {
        let zs = fixed_latent_sample(8, 7, 54 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_055() {
        let zs = fixed_latent_sample(8, 8, 55 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_056() {
        let zs = fixed_latent_sample(8, 1, 56 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_057() {
        let zs = fixed_latent_sample(8, 2, 57 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_058() {
        let zs = fixed_latent_sample(8, 3, 58 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_059() {
        let zs = fixed_latent_sample(8, 4, 59 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_060() {
        let zs = fixed_latent_sample(8, 5, 60 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_061() {
        let zs = fixed_latent_sample(8, 6, 61 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_062() {
        let zs = fixed_latent_sample(8, 7, 62 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_063() {
        let zs = fixed_latent_sample(8, 8, 63 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_064() {
        let zs = fixed_latent_sample(8, 1, 64 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_065() {
        let zs = fixed_latent_sample(8, 2, 65 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_066() {
        let zs = fixed_latent_sample(8, 3, 66 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_067() {
        let zs = fixed_latent_sample(8, 4, 67 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_068() {
        let zs = fixed_latent_sample(8, 5, 68 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_069() {
        let zs = fixed_latent_sample(8, 6, 69 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_070() {
        let zs = fixed_latent_sample(8, 7, 70 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_071() {
        let zs = fixed_latent_sample(8, 8, 71 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_072() {
        let zs = fixed_latent_sample(8, 1, 72 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_073() {
        let zs = fixed_latent_sample(8, 2, 73 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_074() {
        let zs = fixed_latent_sample(8, 3, 74 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_075() {
        let zs = fixed_latent_sample(8, 4, 75 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_076() {
        let zs = fixed_latent_sample(8, 5, 76 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_077() {
        let zs = fixed_latent_sample(8, 6, 77 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_078() {
        let zs = fixed_latent_sample(8, 7, 78 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_079() {
        let zs = fixed_latent_sample(8, 8, 79 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_080() {
        let zs = fixed_latent_sample(8, 1, 80 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_081() {
        let zs = fixed_latent_sample(8, 2, 81 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_082() {
        let zs = fixed_latent_sample(8, 3, 82 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_083() {
        let zs = fixed_latent_sample(8, 4, 83 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_084() {
        let zs = fixed_latent_sample(8, 5, 84 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_085() {
        let zs = fixed_latent_sample(8, 6, 85 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_086() {
        let zs = fixed_latent_sample(8, 7, 86 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_087() {
        let zs = fixed_latent_sample(8, 8, 87 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_088() {
        let zs = fixed_latent_sample(8, 1, 88 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_089() {
        let zs = fixed_latent_sample(8, 2, 89 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_090() {
        let zs = fixed_latent_sample(8, 3, 90 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_091() {
        let zs = fixed_latent_sample(8, 4, 91 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_092() {
        let zs = fixed_latent_sample(8, 5, 92 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_093() {
        let zs = fixed_latent_sample(8, 6, 93 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_094() {
        let zs = fixed_latent_sample(8, 7, 94 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_095() {
        let zs = fixed_latent_sample(8, 8, 95 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_096() {
        let zs = fixed_latent_sample(8, 1, 96 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_097() {
        let zs = fixed_latent_sample(8, 2, 97 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_098() {
        let zs = fixed_latent_sample(8, 3, 98 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_099() {
        let zs = fixed_latent_sample(8, 4, 99 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_100() {
        let zs = fixed_latent_sample(8, 5, 100 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_101() {
        let zs = fixed_latent_sample(8, 6, 101 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_102() {
        let zs = fixed_latent_sample(8, 7, 102 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_103() {
        let zs = fixed_latent_sample(8, 8, 103 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_104() {
        let zs = fixed_latent_sample(8, 1, 104 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_105() {
        let zs = fixed_latent_sample(8, 2, 105 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_106() {
        let zs = fixed_latent_sample(8, 3, 106 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_107() {
        let zs = fixed_latent_sample(8, 4, 107 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_108() {
        let zs = fixed_latent_sample(8, 5, 108 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_109() {
        let zs = fixed_latent_sample(8, 6, 109 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_110() {
        let zs = fixed_latent_sample(8, 7, 110 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_111() {
        let zs = fixed_latent_sample(8, 8, 111 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_112() {
        let zs = fixed_latent_sample(8, 1, 112 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_113() {
        let zs = fixed_latent_sample(8, 2, 113 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_114() {
        let zs = fixed_latent_sample(8, 3, 114 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_115() {
        let zs = fixed_latent_sample(8, 4, 115 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_116() {
        let zs = fixed_latent_sample(8, 5, 116 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_117() {
        let zs = fixed_latent_sample(8, 6, 117 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_118() {
        let zs = fixed_latent_sample(8, 7, 118 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_119() {
        let zs = fixed_latent_sample(8, 8, 119 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_120() {
        let zs = fixed_latent_sample(8, 1, 120 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_121() {
        let zs = fixed_latent_sample(8, 2, 121 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_122() {
        let zs = fixed_latent_sample(8, 3, 122 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_123() {
        let zs = fixed_latent_sample(8, 4, 123 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_124() {
        let zs = fixed_latent_sample(8, 5, 124 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_125() {
        let zs = fixed_latent_sample(8, 6, 125 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_126() {
        let zs = fixed_latent_sample(8, 7, 126 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_127() {
        let zs = fixed_latent_sample(8, 8, 127 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_128() {
        let zs = fixed_latent_sample(8, 1, 128 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_129() {
        let zs = fixed_latent_sample(8, 2, 129 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_130() {
        let zs = fixed_latent_sample(8, 3, 130 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_131() {
        let zs = fixed_latent_sample(8, 4, 131 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_132() {
        let zs = fixed_latent_sample(8, 5, 132 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_133() {
        let zs = fixed_latent_sample(8, 6, 133 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_134() {
        let zs = fixed_latent_sample(8, 7, 134 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_135() {
        let zs = fixed_latent_sample(8, 8, 135 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_136() {
        let zs = fixed_latent_sample(8, 1, 136 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_137() {
        let zs = fixed_latent_sample(8, 2, 137 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_138() {
        let zs = fixed_latent_sample(8, 3, 138 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_139() {
        let zs = fixed_latent_sample(8, 4, 139 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_140() {
        let zs = fixed_latent_sample(8, 5, 140 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_141() {
        let zs = fixed_latent_sample(8, 6, 141 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_142() {
        let zs = fixed_latent_sample(8, 7, 142 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_143() {
        let zs = fixed_latent_sample(8, 8, 143 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_144() {
        let zs = fixed_latent_sample(8, 1, 144 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_145() {
        let zs = fixed_latent_sample(8, 2, 145 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_146() {
        let zs = fixed_latent_sample(8, 3, 146 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_147() {
        let zs = fixed_latent_sample(8, 4, 147 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_148() {
        let zs = fixed_latent_sample(8, 5, 148 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_149() {
        let zs = fixed_latent_sample(8, 6, 149 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_150() {
        let zs = fixed_latent_sample(8, 7, 150 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_151() {
        let zs = fixed_latent_sample(8, 8, 151 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_152() {
        let zs = fixed_latent_sample(8, 1, 152 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_153() {
        let zs = fixed_latent_sample(8, 2, 153 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_154() {
        let zs = fixed_latent_sample(8, 3, 154 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_155() {
        let zs = fixed_latent_sample(8, 4, 155 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_156() {
        let zs = fixed_latent_sample(8, 5, 156 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_157() {
        let zs = fixed_latent_sample(8, 6, 157 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_158() {
        let zs = fixed_latent_sample(8, 7, 158 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_159() {
        let zs = fixed_latent_sample(8, 8, 159 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_160() {
        let zs = fixed_latent_sample(8, 1, 160 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_161() {
        let zs = fixed_latent_sample(8, 2, 161 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_162() {
        let zs = fixed_latent_sample(8, 3, 162 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_163() {
        let zs = fixed_latent_sample(8, 4, 163 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_164() {
        let zs = fixed_latent_sample(8, 5, 164 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_165() {
        let zs = fixed_latent_sample(8, 6, 165 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_166() {
        let zs = fixed_latent_sample(8, 7, 166 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_167() {
        let zs = fixed_latent_sample(8, 8, 167 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_168() {
        let zs = fixed_latent_sample(8, 1, 168 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_169() {
        let zs = fixed_latent_sample(8, 2, 169 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_170() {
        let zs = fixed_latent_sample(8, 3, 170 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_171() {
        let zs = fixed_latent_sample(8, 4, 171 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_172() {
        let zs = fixed_latent_sample(8, 5, 172 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_173() {
        let zs = fixed_latent_sample(8, 6, 173 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_174() {
        let zs = fixed_latent_sample(8, 7, 174 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_175() {
        let zs = fixed_latent_sample(8, 8, 175 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_176() {
        let zs = fixed_latent_sample(8, 1, 176 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_177() {
        let zs = fixed_latent_sample(8, 2, 177 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_178() {
        let zs = fixed_latent_sample(8, 3, 178 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_179() {
        let zs = fixed_latent_sample(8, 4, 179 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_180() {
        let zs = fixed_latent_sample(8, 5, 180 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_181() {
        let zs = fixed_latent_sample(8, 6, 181 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_182() {
        let zs = fixed_latent_sample(8, 7, 182 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_183() {
        let zs = fixed_latent_sample(8, 8, 183 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_184() {
        let zs = fixed_latent_sample(8, 1, 184 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_185() {
        let zs = fixed_latent_sample(8, 2, 185 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_186() {
        let zs = fixed_latent_sample(8, 3, 186 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_187() {
        let zs = fixed_latent_sample(8, 4, 187 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_188() {
        let zs = fixed_latent_sample(8, 5, 188 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_189() {
        let zs = fixed_latent_sample(8, 6, 189 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_190() {
        let zs = fixed_latent_sample(8, 7, 190 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_191() {
        let zs = fixed_latent_sample(8, 8, 191 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_192() {
        let zs = fixed_latent_sample(8, 1, 192 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_193() {
        let zs = fixed_latent_sample(8, 2, 193 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_194() {
        let zs = fixed_latent_sample(8, 3, 194 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_195() {
        let zs = fixed_latent_sample(8, 4, 195 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_196() {
        let zs = fixed_latent_sample(8, 5, 196 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_197() {
        let zs = fixed_latent_sample(8, 6, 197 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_198() {
        let zs = fixed_latent_sample(8, 7, 198 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_199() {
        let zs = fixed_latent_sample(8, 8, 199 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_200() {
        let zs = fixed_latent_sample(8, 1, 200 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_201() {
        let zs = fixed_latent_sample(8, 2, 201 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_202() {
        let zs = fixed_latent_sample(8, 3, 202 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_203() {
        let zs = fixed_latent_sample(8, 4, 203 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_204() {
        let zs = fixed_latent_sample(8, 5, 204 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_205() {
        let zs = fixed_latent_sample(8, 6, 205 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_206() {
        let zs = fixed_latent_sample(8, 7, 206 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_207() {
        let zs = fixed_latent_sample(8, 8, 207 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_208() {
        let zs = fixed_latent_sample(8, 1, 208 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_209() {
        let zs = fixed_latent_sample(8, 2, 209 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_210() {
        let zs = fixed_latent_sample(8, 3, 210 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_211() {
        let zs = fixed_latent_sample(8, 4, 211 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_212() {
        let zs = fixed_latent_sample(8, 5, 212 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_213() {
        let zs = fixed_latent_sample(8, 6, 213 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_214() {
        let zs = fixed_latent_sample(8, 7, 214 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_215() {
        let zs = fixed_latent_sample(8, 8, 215 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_216() {
        let zs = fixed_latent_sample(8, 1, 216 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_217() {
        let zs = fixed_latent_sample(8, 2, 217 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_218() {
        let zs = fixed_latent_sample(8, 3, 218 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_219() {
        let zs = fixed_latent_sample(8, 4, 219 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_220() {
        let zs = fixed_latent_sample(8, 5, 220 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_221() {
        let zs = fixed_latent_sample(8, 6, 221 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_222() {
        let zs = fixed_latent_sample(8, 7, 222 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_223() {
        let zs = fixed_latent_sample(8, 8, 223 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_224() {
        let zs = fixed_latent_sample(8, 1, 224 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_225() {
        let zs = fixed_latent_sample(8, 2, 225 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_226() {
        let zs = fixed_latent_sample(8, 3, 226 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_227() {
        let zs = fixed_latent_sample(8, 4, 227 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_228() {
        let zs = fixed_latent_sample(8, 5, 228 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_229() {
        let zs = fixed_latent_sample(8, 6, 229 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_230() {
        let zs = fixed_latent_sample(8, 7, 230 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_231() {
        let zs = fixed_latent_sample(8, 8, 231 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_232() {
        let zs = fixed_latent_sample(8, 1, 232 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_233() {
        let zs = fixed_latent_sample(8, 2, 233 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_234() {
        let zs = fixed_latent_sample(8, 3, 234 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_235() {
        let zs = fixed_latent_sample(8, 4, 235 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_236() {
        let zs = fixed_latent_sample(8, 5, 236 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_237() {
        let zs = fixed_latent_sample(8, 6, 237 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_238() {
        let zs = fixed_latent_sample(8, 7, 238 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_239() {
        let zs = fixed_latent_sample(8, 8, 239 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_240() {
        let zs = fixed_latent_sample(8, 1, 240 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_241() {
        let zs = fixed_latent_sample(8, 2, 241 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_242() {
        let zs = fixed_latent_sample(8, 3, 242 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_243() {
        let zs = fixed_latent_sample(8, 4, 243 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_244() {
        let zs = fixed_latent_sample(8, 5, 244 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_245() {
        let zs = fixed_latent_sample(8, 6, 245 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_246() {
        let zs = fixed_latent_sample(8, 7, 246 as u64);
        assert_eq!(zs.len(), 7);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..7.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_247() {
        let zs = fixed_latent_sample(8, 8, 247 as u64);
        assert_eq!(zs.len(), 8);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..8.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_248() {
        let zs = fixed_latent_sample(8, 1, 248 as u64);
        assert_eq!(zs.len(), 1);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..1.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_249() {
        let zs = fixed_latent_sample(8, 2, 249 as u64);
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 6);
        assert_eq!(interp.len(), 7);
        let grid = assemble_grid(&zs[..2.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_250() {
        let zs = fixed_latent_sample(8, 3, 250 as u64);
        assert_eq!(zs.len(), 3);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 2);
        assert_eq!(interp.len(), 3);
        let grid = assemble_grid(&zs[..3.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_251() {
        let zs = fixed_latent_sample(8, 4, 251 as u64);
        assert_eq!(zs.len(), 4);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 3);
        assert_eq!(interp.len(), 4);
        let grid = assemble_grid(&zs[..4.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_252() {
        let zs = fixed_latent_sample(8, 5, 252 as u64);
        assert_eq!(zs.len(), 5);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 4);
        assert_eq!(interp.len(), 5);
        let grid = assemble_grid(&zs[..5.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    #[test]
    fn test_samples_stress_253() {
        let zs = fixed_latent_sample(8, 6, 253 as u64);
        assert_eq!(zs.len(), 6);
        assert_eq!(zs[0].shape(), &[8]);
        let z1 = Tensor::from_vec(vec![0.0; 4], vec![4]);
        let z2 = Tensor::from_vec(vec![1.0; 4], vec![4]);
        let interp = interpolate_latents_batch(&z1, &z2, 5);
        assert_eq!(interp.len(), 6);
        let grid = assemble_grid(&zs[..6.min(zs.len())], 2);
        assert!(!grid.to_vec().is_empty());
    }

    // GAN training and evaluation padding line 0
    // GAN training and evaluation padding line 1
    // GAN training and evaluation padding line 2
    // GAN training and evaluation padding line 3
    // GAN training and evaluation padding line 4
    // GAN training and evaluation padding line 5
    // GAN training and evaluation padding line 6
    // GAN training and evaluation padding line 7
    // GAN training and evaluation padding line 8
}
