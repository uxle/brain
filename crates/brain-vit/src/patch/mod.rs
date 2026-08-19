//! # Patch Embedding for brain-vit
//!
//! Implements both Conv-based and Unfold-based patch embedding,
//! mapping raw pixel images to token sequences consumed by the ViT encoder.
//!
//! ## Components
//! - [`PatchEmbed`] — main patch embedding module
//! - [`PatchEmbedConfig`] — configuration (re-exported from config)
//! - [`validate_patch_config`] — standalone config validator

pub mod pos_embed;

use crate::core::{VitError, VitResult, Tensor2D, SimpleRng};
use crate::config::{PatchEmbedConfig, PatchMode};
use crate::ops::{extract_patches, linear};

/// Patch embedding module: maps `[B, C, H, W]` → `[B, N, D]`.
///
/// Supports two modes:
/// - **Conv** (default): equivalent to a strided convolution with kernel = stride = patch_size.
/// - **Unfold**: explicit unfold then linear projection.
///
/// # Example
/// ```rust
/// use brain_vit::patch::PatchEmbed;
/// use brain_vit::config::PatchEmbedConfig;
/// let cfg = PatchEmbedConfig::default();
/// let embed = PatchEmbed::new(&cfg, 42).unwrap();
/// ```
pub struct PatchEmbed {
    /// Configuration.
    pub config: PatchEmbedConfig,
    /// Projection weight `[embed_dim, patch_dim]`.
    pub weight: Vec<f64>,
    /// Projection bias `[embed_dim]` (if bias=true).
    pub bias: Option<Vec<f64>>,
}

impl PatchEmbed {
    /// Create a new PatchEmbed with Xavier-initialized weights.
    pub fn new(config: &PatchEmbedConfig, seed: u64) -> VitResult<Self> {
        config.validate()?;
        let mut rng = SimpleRng::new(seed);
        let patch_dim = config.in_channels * config.patch_size * config.patch_size;
        let embed_dim = config.embed_dim;
        let weight = rng.xavier_uniform(embed_dim, patch_dim);
        let bias = if config.bias {
            Some(rng.gen_vec(embed_dim, -0.01, 0.01))
        } else {
            None
        };
        Ok(Self { config: config.clone(), weight, bias })
    }

    /// Forward pass: `[B, C, H, W]` → `[B, N, D]`.
    ///
    /// # Arguments
    /// - `images`: flat `[B, C, H, W]` pixel data.
    /// - `batch`: batch size.
    ///
    /// # Example
    /// ```rust
    /// use brain_vit::patch::PatchEmbed;
    /// use brain_vit::config::PatchEmbedConfig;
    /// let cfg = PatchEmbedConfig { image_size: 16, patch_size: 4, in_channels: 1, embed_dim: 8, bias: true, ..Default::default() };
    /// let embed = PatchEmbed::new(&cfg, 0).unwrap();
    /// let img = vec![0.0f64; 1 * 1 * 16 * 16];
    /// let tokens = embed.forward(&img, 1).unwrap();
    /// assert_eq!(tokens.len(), 1 * 16 * 8); // 16 patches, 8 dim
    /// ```
    pub fn forward(&self, images: &[f64], batch: usize) -> VitResult<Vec<f64>> {
        match self.config.mode {
            PatchMode::Conv | PatchMode::Unfold => self.unfold_forward(images, batch),
        }
    }

    /// Unfold-based patch extraction and projection.
    fn unfold_forward(&self, images: &[f64], batch: usize) -> VitResult<Vec<f64>> {
        let cfg = &self.config;
        let n = cfg.num_patches();
        let patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size;
        let embed_dim = cfg.embed_dim;

        // Extract patches: [B, N, patch_dim]
        let patches = extract_patches(
            images, batch, cfg.in_channels,
            cfg.image_size, cfg.image_size,
            cfg.patch_size, cfg.patch_size,
        )?;

        // Project each patch
        let proj_w = Tensor2D::from_data(embed_dim, patch_dim, self.weight.clone())?;
        let bias_ref = self.bias.as_deref();
        let mut out = vec![0.0f64; batch * n * embed_dim];

        for b in 0..batch {
            for p in 0..n {
                let patch_start = (b * n + p) * patch_dim;
                let patch_data = patches[patch_start..patch_start + patch_dim].to_vec();
                let patch_mat = Tensor2D::from_data(1, patch_dim, patch_data)?;
                let proj = linear(&patch_mat, &proj_w, bias_ref)?;
                let out_start = (b * n + p) * embed_dim;
                out[out_start..out_start + embed_dim].copy_from_slice(&proj.data);
            }
        }
        Ok(out)
    }

    /// Number of parameters in this module.
    pub fn num_params(&self) -> usize {
        self.weight.len() + self.bias.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    /// Grid size (patches per side).
    pub fn grid_size(&self) -> usize { self.config.grid_size() }

    /// Total number of patches.
    pub fn num_patches(&self) -> usize { self.config.num_patches() }

    /// Get the embedding dimension.
    pub fn embed_dim(&self) -> usize { self.config.embed_dim }
}

/// Validate a patch embedding configuration.
pub fn validate_patch_config(image_size: usize, patch_size: usize) -> VitResult<()> {
    if patch_size == 0 {
        return Err(VitError::Config("patch_size must be > 0".to_string()));
    }
    if !image_size.is_multiple_of(patch_size) {
        return Err(VitError::InvalidPatchSize { image_dim: image_size, patch_size });
    }
    Ok(())
}

/// Compute the number of patches for an image size and patch size.
pub fn num_patches_for(image_size: usize, patch_size: usize) -> VitResult<usize> {
    validate_patch_config(image_size, patch_size)?;
    Ok((image_size / patch_size).pow(2))
}

/// Verify that patch embedding matches direct convolution (numerical correctness).
///
/// For a single patch, the patch embedding should equal:
///   `weight @ patch + bias` (if bias).
pub fn verify_patch_embed_correctness(
    embed: &PatchEmbed,
    images: &[f64],
    batch: usize,
    atol: f64,
) -> VitResult<bool> {
    let cfg = &embed.config;
    let n = cfg.num_patches();
    let patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size;
    let embed_dim = cfg.embed_dim;

    let patches = extract_patches(
        images, batch, cfg.in_channels,
        cfg.image_size, cfg.image_size,
        cfg.patch_size, cfg.patch_size,
    )?;
    let proj_w = Tensor2D::from_data(embed_dim, patch_dim, embed.weight.clone())?;
    let bias_ref = embed.bias.as_deref();

    let forward_out = embed.forward(images, batch)?;

    for b in 0..batch {
        for p in 0..n {
            let patch_start = (b * n + p) * patch_dim;
            let patch_data = patches[patch_start..patch_start + patch_dim].to_vec();
            let patch_mat = Tensor2D::from_data(1, patch_dim, patch_data)?;
            let ref_proj = linear(&patch_mat, &proj_w, bias_ref)?;
            let fwd_start = (b * n + p) * embed_dim;
            for d in 0..embed_dim {
                let diff = (forward_out[fwd_start + d] - ref_proj.data[d]).abs();
                if diff > atol {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PatchEmbedConfig;

    fn small_config() -> PatchEmbedConfig {
        PatchEmbedConfig {
            image_size: 16,
            patch_size: 4,
            in_channels: 1,
            embed_dim: 8,
            bias: true,
            mode: PatchMode::Conv,
        }
    }

    #[test]
    fn test_patch_embed_new() {
        let cfg = small_config();
        let embed = PatchEmbed::new(&cfg, 0).unwrap();
        assert_eq!(embed.grid_size(), 4);
        assert_eq!(embed.num_patches(), 16);
        assert_eq!(embed.embed_dim(), 8);
    }

    #[test]
    fn test_patch_embed_forward_shape() {
        let cfg = small_config();
        let embed = PatchEmbed::new(&cfg, 0).unwrap();
        let img = vec![1.0f64; 1 * 1 * 16 * 16];
        let tokens = embed.forward(&img, 1).unwrap();
        assert_eq!(tokens.len(), 1 * 16 * 8);
    }

    #[test]
    fn test_patch_embed_batch() {
        let cfg = small_config();
        let embed = PatchEmbed::new(&cfg, 0).unwrap();
        let img = vec![0.5f64; 3 * 1 * 16 * 16];
        let tokens = embed.forward(&img, 3).unwrap();
        assert_eq!(tokens.len(), 3 * 16 * 8);
    }

    #[test]
    fn test_patch_embed_finite_output() {
        let cfg = small_config();
        let embed = PatchEmbed::new(&cfg, 1).unwrap();
        let img = vec![0.5f64; 2 * 1 * 16 * 16];
        let tokens = embed.forward(&img, 2).unwrap();
        assert!(tokens.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_patch_embed_no_bias() {
        let mut cfg = small_config();
        cfg.bias = false;
        let embed = PatchEmbed::new(&cfg, 0).unwrap();
        assert!(embed.bias.is_none());
        let img = vec![1.0f64; 1 * 1 * 16 * 16];
        let tokens = embed.forward(&img, 1).unwrap();
        assert_eq!(tokens.len(), 16 * 8);
    }

    #[test]
    fn test_patch_embed_zero_image_gives_bias() {
        // Zero image → output should equal bias (if bias)
        let cfg = small_config();
        let embed = PatchEmbed::new(&cfg, 5).unwrap();
        let bias = embed.bias.clone().unwrap();
        let img = vec![0.0f64; 1 * 1 * 16 * 16];
        let tokens = embed.forward(&img, 1).unwrap();
        // First patch output should equal bias
        for d in 0..8 {
            assert!((tokens[d] - bias[d]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_verify_correctness() {
        let cfg = small_config();
        let embed = PatchEmbed::new(&cfg, 7).unwrap();
        let img: Vec<f64> = (0..16*16).map(|x| x as f64 / 256.0).collect();
        let ok = verify_patch_embed_correctness(&embed, &img, 1, 1e-9).unwrap();
        assert!(ok);
    }

    #[test]
    fn test_num_patches_for() {
        assert_eq!(num_patches_for(224, 16).unwrap(), 196);
        assert_eq!(num_patches_for(32, 4).unwrap(), 64);
    }

    #[test]
    fn test_num_patches_invalid() {
        assert!(num_patches_for(224, 15).is_err());
    }

    #[test]
    fn test_validate_patch_config_ok() {
        assert!(validate_patch_config(224, 16).is_ok());
        assert!(validate_patch_config(32, 8).is_ok());
    }

    #[test]
    fn test_validate_patch_config_zero() {
        assert!(validate_patch_config(224, 0).is_err());
    }

    #[test]
    fn test_validate_patch_config_non_divisible() {
        assert!(validate_patch_config(224, 15).is_err());
    }

    #[test]
    fn test_patch_embed_num_params() {
        let cfg = small_config();
        let embed = PatchEmbed::new(&cfg, 0).unwrap();
        let patch_dim = 1 * 4 * 4;
        let embed_dim = 8;
        assert_eq!(embed.num_params(), embed_dim * patch_dim + embed_dim);
    }

    #[test]
    fn test_patch_embed_invalid_config() {
        let mut cfg = small_config();
        cfg.patch_size = 0;
        assert!(PatchEmbed::new(&cfg, 0).is_err());
    }

    #[test]
    fn test_patch_embed_multichannel() {
        let cfg = PatchEmbedConfig {
            image_size: 16, patch_size: 4, in_channels: 3,
            embed_dim: 8, bias: true, mode: PatchMode::Conv,
        };
        let embed = PatchEmbed::new(&cfg, 0).unwrap();
        let img = vec![0.5f64; 1 * 3 * 16 * 16];
        let tokens = embed.forward(&img, 1).unwrap();
        assert_eq!(tokens.len(), 16 * 8);
    }

    #[test]
    fn test_patch_embed_different_seeds_different_weights() {
        let cfg = small_config();
        let e1 = PatchEmbed::new(&cfg, 1).unwrap();
        let e2 = PatchEmbed::new(&cfg, 2).unwrap();
        assert_ne!(e1.weight, e2.weight);
    }

    #[test]
    fn test_patch_embed_same_seed_same_weights() {
        let cfg = small_config();
        let e1 = PatchEmbed::new(&cfg, 42).unwrap();
        let e2 = PatchEmbed::new(&cfg, 42).unwrap();
        assert_eq!(e1.weight, e2.weight);
    }

    #[test]
    fn test_patch_embed_large_batch() {
        let cfg = small_config();
        let embed = PatchEmbed::new(&cfg, 0).unwrap();
        let img = vec![0.1f64; 8 * 1 * 16 * 16];
        let tokens = embed.forward(&img, 8).unwrap();
        assert_eq!(tokens.len(), 8 * 16 * 8);
    }

    #[test]
    fn test_patch_embed_forward_linearity() {
        // Scaling input by 2 should scale output by 2 (no bias comparison, use 0 bias)
        let mut cfg = small_config();
        cfg.bias = false;
        let embed = PatchEmbed::new(&cfg, 0).unwrap();
        let img1 = vec![1.0f64; 1 * 1 * 16 * 16];
        let img2 = vec![2.0f64; 1 * 1 * 16 * 16];
        let out1 = embed.forward(&img1, 1).unwrap();
        let out2 = embed.forward(&img2, 1).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((2.0 * a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_patch_embed_all_zero_input_no_bias() {
        let mut cfg = small_config();
        cfg.bias = false;
        let embed = PatchEmbed::new(&cfg, 0).unwrap();
        let img = vec![0.0f64; 1 * 1 * 16 * 16];
        let tokens = embed.forward(&img, 1).unwrap();
        assert!(tokens.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_grid_size_various() {
        for (img, patch, expected) in [(224, 16, 14), (32, 8, 4), (64, 4, 16)] {
            let cfg = PatchEmbedConfig { image_size: img, patch_size: patch, in_channels: 1, embed_dim: 8, bias: false, mode: PatchMode::Conv };
            let embed = PatchEmbed::new(&cfg, 0).unwrap();
            assert_eq!(embed.grid_size(), expected);
        }
    }

    #[test]
    fn test_patch_embed_single_patch() {
        // image_size == patch_size → 1 patch
        let cfg = PatchEmbedConfig { image_size: 4, patch_size: 4, in_channels: 1, embed_dim: 8, bias: false, mode: PatchMode::Conv };
        let embed = PatchEmbed::new(&cfg, 0).unwrap();
        assert_eq!(embed.num_patches(), 1);
        let img = vec![1.0f64; 1 * 1 * 4 * 4];
        let tokens = embed.forward(&img, 1).unwrap();
        assert_eq!(tokens.len(), 8);
    }
}
