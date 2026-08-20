//! # Position Embeddings for brain-vit
//!
//! Supports three position embedding strategies:
//! - **Learned 1D**: trainable `[seq_len, embed_dim]` embedding table
//! - **Learned 2D**: trainable grid `[H, W, embed_dim]` bicubic-interpolatable
//! - **Sinusoidal**: fixed frequency encoding from the original Transformer
//!
//! Position embeddings are added to patch tokens after projection.

use crate::config::{PosEmbedConfig, PosEmbedType};
use crate::core::{SimpleRng, VitError, VitResult};
use crate::utils::interpolate_2d;

/// Position embedding module.
///
/// # Example
/// ```rust
/// use brain_vit::patch::pos_embed::{PosEmbed, PosEmbedBuilder};
/// use brain_vit::config::{PosEmbedConfig, PosEmbedType};
/// let cfg = PosEmbedConfig { seq_len: 197, embed_dim: 768, ..Default::default() };
/// let pe = PosEmbed::new(&cfg, 0).unwrap();
/// assert_eq!(pe.embed.len(), 197 * 768);
/// ```
pub struct PosEmbed {
    /// Embedding table `[seq_len, embed_dim]`.
    pub embed: Vec<f64>,
    /// Configuration.
    pub config: PosEmbedConfig,
}

impl PosEmbed {
    /// Create a new position embedding from config.
    pub fn new(config: &PosEmbedConfig, seed: u64) -> VitResult<Self> {
        config.validate()?;
        let embed = match config.embed_type {
            PosEmbedType::Learned1D => {
                let mut rng = SimpleRng::new(seed);
                rng.gen_vec(config.seq_len * config.embed_dim, -0.02, 0.02)
            }
            PosEmbedType::Learned2D => {
                let mut rng = SimpleRng::new(seed);
                rng.gen_vec(config.seq_len * config.embed_dim, -0.02, 0.02)
            }
            PosEmbedType::Sinusoidal => sinusoidal_1d(config.seq_len, config.embed_dim),
            PosEmbedType::None => {
                vec![0.0f64; config.seq_len * config.embed_dim]
            }
        };
        Ok(Self {
            embed,
            config: config.clone(),
        })
    }

    /// Add position embedding to token sequence in-place.
    ///
    /// - `tokens`: `[B, seq_len, embed_dim]` flat.
    /// - `batch`: batch size.
    pub fn add_to(&self, tokens: &mut [f64], batch: usize) -> VitResult<()> {
        let seq_len = self.config.seq_len;
        let embed_dim = self.config.embed_dim;
        let expected = batch * seq_len * embed_dim;
        if tokens.len() != expected {
            return Err(VitError::Shape(format!(
                "PosEmbed::add_to: expected {} elements, got {}",
                expected,
                tokens.len()
            )));
        }
        for b in 0..batch {
            let base = b * seq_len * embed_dim;
            for s in 0..seq_len {
                for d in 0..embed_dim {
                    tokens[base + s * embed_dim + d] += self.embed[s * embed_dim + d];
                }
            }
        }
        Ok(())
    }

    /// Interpolate position embedding to a new grid size.
    ///
    /// Used when fine-tuning at a different resolution than pretraining.
    /// The CLS token embedding is kept unchanged; only patch positions are interpolated.
    ///
    /// # Arguments
    /// - `new_h`, `new_w`: new grid height and width.
    ///
    /// # Returns
    /// New `PosEmbed` with interpolated position embeddings.
    pub fn interpolate_to(&self, new_h: usize, new_w: usize) -> VitResult<PosEmbed> {
        let embed_dim = self.config.embed_dim;
        let old_h = self.config.grid_h;
        let old_w = self.config.grid_w;
        let has_cls = self.config.has_cls_token;

        let cls_offset = if has_cls { embed_dim } else { 0 };
        let old_n = old_h * old_w;
        let new_n = new_h * new_w;

        // Extract CLS embedding (if present)
        let cls_part = if has_cls {
            self.embed[..cls_offset].to_vec()
        } else {
            vec![]
        };

        // Reshape patch embeddings: [old_n, embed_dim] → interpolate per-dim
        let patch_embed = &self.embed[cls_offset..cls_offset + old_n * embed_dim];

        // Interpolate each embedding dimension independently
        let mut new_patch_embed = vec![0.0f64; new_n * embed_dim];
        for d in 0..embed_dim {
            // Extract grid for this dimension: [old_h, old_w]
            let grid: Vec<f64> = (0..old_n).map(|p| patch_embed[p * embed_dim + d]).collect();
            let interp = interpolate_2d(&grid, old_h, old_w, new_h, new_w)?;
            for p in 0..new_n {
                new_patch_embed[p * embed_dim + d] = interp[p];
            }
        }

        let new_embed = [cls_part, new_patch_embed].concat();

        let mut new_config = self.config.clone();
        new_config.seq_len = new_n + if has_cls { 1 } else { 0 };
        new_config.grid_h = new_h;
        new_config.grid_w = new_w;

        Ok(PosEmbed {
            embed: new_embed,
            config: new_config,
        })
    }

    /// Number of position embedding vectors.
    pub fn seq_len(&self) -> usize {
        self.config.seq_len
    }

    /// Embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// Get the embedding vector for a specific position.
    pub fn get_pos(&self, pos: usize) -> &[f64] {
        let d = self.config.embed_dim;
        &self.embed[pos * d..(pos + 1) * d]
    }

    /// Norm of a position embedding vector.
    pub fn pos_norm(&self, pos: usize) -> f64 {
        let v = self.get_pos(pos);
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

/// Generate sinusoidal 1D position encoding `[seq_len, embed_dim]`.
pub fn sinusoidal_1d(seq_len: usize, embed_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; seq_len * embed_dim];
    for pos in 0..seq_len {
        for i in 0..embed_dim {
            let angle = pos as f64 / 10000f64.powf(2.0 * (i / 2) as f64 / embed_dim as f64);
            out[pos * embed_dim + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    out
}

/// Generate 2D sinusoidal position encoding for a grid.
///
/// Returns `[H * W, embed_dim]` flat.
pub fn sinusoidal_2d(grid_h: usize, grid_w: usize, embed_dim: usize) -> Vec<f64> {
    let n = grid_h * grid_w;
    let d_half = embed_dim / 2;
    let mut out = vec![0.0f64; n * embed_dim];
    for h in 0..grid_h {
        for w in 0..grid_w {
            let pos = h * grid_w + w;
            for i in 0..d_half {
                let omega = 1.0 / 10000f64.powf(2.0 * i as f64 / embed_dim as f64);
                let angle_h = h as f64 * omega;
                let angle_w = w as f64 * omega;
                out[pos * embed_dim + 2 * i] = angle_h.sin();
                out[pos * embed_dim + 2 * i + 1] = angle_w.cos();
            }
        }
    }
    out
}

/// Builder for PosEmbed with fluent configuration.
pub struct PosEmbedBuilder {
    config: PosEmbedConfig,
    seed: u64,
}

impl PosEmbedBuilder {
    /// Create builder with defaults.
    pub fn new() -> Self {
        Self {
            config: PosEmbedConfig::default(),
            seed: 0,
        }
    }

    /// Set sequence length.
    pub fn seq_len(mut self, n: usize) -> Self {
        self.config.seq_len = n;
        self
    }

    /// Set embedding dimension.
    pub fn embed_dim(mut self, d: usize) -> Self {
        self.config.embed_dim = d;
        self
    }

    /// Set embed type.
    pub fn embed_type(mut self, t: PosEmbedType) -> Self {
        self.config.embed_type = t;
        self
    }

    /// Set grid shape for 2D embeddings.
    pub fn grid(mut self, h: usize, w: usize) -> Self {
        self.config.grid_h = h;
        self.config.grid_w = w;
        self
    }

    /// Set random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Set whether CLS token is present.
    pub fn has_cls_token(mut self, v: bool) -> Self {
        self.config.has_cls_token = v;
        self
    }

    /// Build the PosEmbed.
    pub fn build(self) -> VitResult<PosEmbed> {
        PosEmbed::new(&self.config, self.seed)
    }
}

impl Default for PosEmbedBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Check that sinusoidal encoding is unique per position (no duplicates).
pub fn check_sinusoidal_uniqueness(seq_len: usize, embed_dim: usize) -> bool {
    let enc = sinusoidal_1d(seq_len, embed_dim);
    for i in 0..seq_len {
        for j in 0..seq_len {
            if i == j {
                continue;
            }
            let a = &enc[i * embed_dim..(i + 1) * embed_dim];
            let b = &enc[j * embed_dim..(j + 1) * embed_dim];
            if a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-10) {
                return false;
            }
        }
    }
    true
}

/// Verify that interpolated pos embeddings preserve values at grid points.
pub fn verify_interpolation_preserves_corners(pe: &PosEmbed) -> VitResult<bool> {
    let new_pe = pe.interpolate_to(pe.config.grid_h, pe.config.grid_w)?;
    let tol = 1e-7;
    let has_cls = pe.config.has_cls_token;
    let cls_off = if has_cls { pe.config.embed_dim } else { 0 };
    let n = pe.config.grid_h * pe.config.grid_w;
    for p in 0..n {
        for d in 0..pe.config.embed_dim {
            let a = pe.embed[cls_off + p * pe.config.embed_dim + d];
            let b = new_pe.embed[cls_off + p * pe.config.embed_dim + d];
            if (a - b).abs() > tol {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PosEmbedConfig, PosEmbedType};

    fn small_config() -> PosEmbedConfig {
        PosEmbedConfig {
            seq_len: 17, // 4x4 patches + CLS
            embed_dim: 8,
            embed_type: PosEmbedType::Learned1D,
            has_cls_token: true,
            grid_h: 4,
            grid_w: 4,
            dropout: 0.0,
        }
    }

    #[test]
    fn test_pos_embed_learned_1d() {
        let cfg = small_config();
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        assert_eq!(pe.embed.len(), 17 * 8);
        assert_eq!(pe.seq_len(), 17);
        assert_eq!(pe.embed_dim(), 8);
    }

    #[test]
    fn test_pos_embed_sinusoidal() {
        let mut cfg = small_config();
        cfg.embed_type = PosEmbedType::Sinusoidal;
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        assert_eq!(pe.embed.len(), 17 * 8);
        // All values in [-1, 1]
        assert!(pe.embed.iter().all(|&v| v >= -1.0 && v <= 1.0));
    }

    #[test]
    fn test_pos_embed_none() {
        let mut cfg = small_config();
        cfg.embed_type = PosEmbedType::None;
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        assert!(pe.embed.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_pos_embed_add_to() {
        let cfg = small_config();
        let pe = PosEmbed::new(&cfg, 1).unwrap();
        let mut tokens = vec![0.0f64; 2 * 17 * 8];
        pe.add_to(&mut tokens, 2).unwrap();
        // Tokens should now contain pe values
        for s in 0..17 {
            for d in 0..8 {
                assert!((tokens[s * 8 + d] - pe.embed[s * 8 + d]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_pos_embed_add_to_shape_err() {
        let cfg = small_config();
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        let mut bad_tokens = vec![0.0f64; 10]; // wrong size
        assert!(pe.add_to(&mut bad_tokens, 1).is_err());
    }

    #[test]
    fn test_pos_embed_add_broadcast() {
        // Adding to two samples should be the same as adding to one
        let cfg = small_config();
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        let mut t1 = vec![1.0f64; 17 * 8];
        let mut t2 = vec![1.0f64; 2 * 17 * 8];
        pe.add_to(&mut t1, 1).unwrap();
        pe.add_to(&mut t2, 2).unwrap();
        for (a, b) in t1.iter().zip(t2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_interpolate_same_size() {
        let cfg = small_config();
        let pe = PosEmbed::new(&cfg, 42).unwrap();
        let ok = verify_interpolation_preserves_corners(&pe).unwrap();
        assert!(ok, "Interpolation at same size should preserve values");
    }

    #[test]
    fn test_interpolate_upsample() {
        let cfg = small_config();
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        let new_pe = pe.interpolate_to(8, 8).unwrap(); // 4x4 → 8x8
                                                       // New seq_len = 64 patches + 1 CLS = 65
        assert_eq!(new_pe.seq_len(), 65);
        assert_eq!(new_pe.embed.len(), 65 * 8);
    }

    #[test]
    fn test_interpolate_cls_preserved() {
        let cfg = small_config();
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        let new_pe = pe.interpolate_to(8, 8).unwrap();
        // CLS token (first embed_dim values) should match original
        for d in 0..8 {
            assert!((pe.embed[d] - new_pe.embed[d]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_sinusoidal_1d_shape() {
        let enc = sinusoidal_1d(10, 64);
        assert_eq!(enc.len(), 10 * 64);
    }

    #[test]
    fn test_sinusoidal_1d_bounds() {
        let enc = sinusoidal_1d(50, 128);
        assert!(enc.iter().all(|&v| v >= -1.0 && v <= 1.0));
    }

    #[test]
    fn test_sinusoidal_uniqueness() {
        assert!(check_sinusoidal_uniqueness(10, 16));
    }

    #[test]
    fn test_sinusoidal_2d_shape() {
        let enc = sinusoidal_2d(7, 7, 64);
        assert_eq!(enc.len(), 49 * 64);
    }

    #[test]
    fn test_sinusoidal_2d_bounds() {
        let enc = sinusoidal_2d(4, 4, 32);
        for &v in &enc {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_pos_embed_builder() {
        let pe = PosEmbedBuilder::new()
            .seq_len(10)
            .embed_dim(32)
            .embed_type(PosEmbedType::Sinusoidal)
            .seed(99)
            .build()
            .unwrap();
        assert_eq!(pe.seq_len(), 10);
        assert_eq!(pe.embed_dim(), 32);
    }

    #[test]
    fn test_pos_embed_get_pos() {
        let cfg = small_config();
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        let v = pe.get_pos(3);
        assert_eq!(v.len(), 8);
        // Should equal embed[3*8..(3+1)*8]
        for (a, b) in v.iter().zip(pe.embed[24..32].iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_pos_embed_norm() {
        let mut cfg = small_config();
        cfg.embed_type = PosEmbedType::Sinusoidal;
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        let n = pe.pos_norm(0);
        assert!(n.is_finite() && n >= 0.0);
    }

    #[test]
    fn test_pos_embed_learned_2d() {
        let mut cfg = small_config();
        cfg.embed_type = PosEmbedType::Learned2D;
        let pe = PosEmbed::new(&cfg, 3).unwrap();
        assert_eq!(pe.embed.len(), 17 * 8);
    }

    #[test]
    fn test_pos_embed_same_seed_deterministic() {
        let cfg = small_config();
        let p1 = PosEmbed::new(&cfg, 77).unwrap();
        let p2 = PosEmbed::new(&cfg, 77).unwrap();
        assert_eq!(p1.embed, p2.embed);
    }

    #[test]
    fn test_pos_embed_different_seeds_differ() {
        let cfg = small_config();
        let p1 = PosEmbed::new(&cfg, 1).unwrap();
        let p2 = PosEmbed::new(&cfg, 2).unwrap();
        assert_ne!(p1.embed, p2.embed);
    }

    #[test]
    fn test_sinusoidal_pos0() {
        // Position 0 even dims should be sin(0)=0, odd dims cos(0)=1
        let enc = sinusoidal_1d(5, 4);
        assert!(enc[0].abs() < 1e-9); // sin(0)=0
        assert!((enc[1] - 1.0).abs() < 1e-9); // cos(0)=1
    }

    #[test]
    fn test_pos_embed_add_to_idempotent_zero_embed() {
        let mut cfg = small_config();
        cfg.embed_type = PosEmbedType::None;
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        let original = vec![1.5f64; 1 * 17 * 8];
        let mut tokens = original.clone();
        pe.add_to(&mut tokens, 1).unwrap();
        assert_eq!(tokens, original);
    }

    #[test]
    fn test_builder_default() {
        let builder = PosEmbedBuilder::default();
        let pe = builder.build().unwrap();
        assert_eq!(pe.seq_len(), 197); // default
    }
}
