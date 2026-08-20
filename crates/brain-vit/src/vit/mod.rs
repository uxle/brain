//! # ViT Model for brain-vit
//!
//! The top-level `ViT` model: patch embedding → CLS prepend → position embed
//! → transformer encoder (N blocks) → pooling → classification head.
//!
//! Designed for easy composition: use [`VitBackbone`] when you need features
//! without a head, or attach any [`VitHead`] variant for task-specific output.

pub mod blocks;
pub mod heads;

use crate::config::{PosEmbedConfig, VitBlockConfig, VitConfig};
use crate::core::{SimpleRng, Tensor3D, VitError, VitOutput, VitResult, VitState};
use crate::patch::pos_embed::PosEmbed;
use crate::patch::PatchEmbed;
use crate::vit::blocks::VitBlock;
use crate::vit::heads::{ClsHead, VitHead};
use std::collections::HashMap;

/// Full Vision Transformer model.
///
/// Assembles:
/// 1. Patch embedding [`PatchEmbed`]
/// 2. CLS token (optional)
/// 3. Position embedding [`PosEmbed`]
/// 4. N × [`VitBlock`]
/// 5. Head (classification, projection, etc.)
///
/// # Example
/// ```rust
/// use brain_vit::vit::ViT;
/// use brain_vit::config::VitConfig;
/// let cfg = VitConfig::tiny();
/// // Note: tiny config uses image_size=224 which needs adjustment for fast tests
/// ```
pub struct ViT {
    /// Configuration.
    pub config: VitConfig,
    /// Patch embedding module.
    pub patch_embed: PatchEmbed,
    /// CLS token vector `[embed_dim]`.
    pub cls_token: Vec<f64>,
    /// Position embedding module.
    pub pos_embed: PosEmbed,
    /// Transformer blocks.
    pub blocks: Vec<VitBlock>,
    /// Classification head.
    pub head: ClsHead,
    /// Runtime state.
    pub state: VitState,
}

impl ViT {
    /// Create a new ViT model with random initialization.
    ///
    /// # Arguments
    /// - `config`: Full ViT configuration.
    /// - `seed`: Random seed for weight initialization.
    pub fn new(config: VitConfig, seed: u64) -> VitResult<Self> {
        config.validate()?;
        let mut rng = SimpleRng::new(seed);

        let patch_embed = PatchEmbed::new(
            &config.patch_embed,
            rng.next_usize(u32::MAX as usize) as u64,
        )?;
        let embed_dim = config.embed_dim();
        let cls_token = rng.gen_vec(embed_dim, -0.02, 0.02);

        let pos_embed_cfg = PosEmbedConfig {
            seq_len: config.seq_len(),
            embed_dim,
            embed_type: config.pos_embed.embed_type,
            has_cls_token: config.use_cls_token,
            grid_h: config.patch_embed.grid_size(),
            grid_w: config.patch_embed.grid_size(),
            dropout: config.pos_embed.dropout,
        };
        let pos_embed = PosEmbed::new(&pos_embed_cfg, rng.next_usize(u32::MAX as usize) as u64)?;

        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let drop_path = config.block_drop_path_rate(i);
            let block_cfg = VitBlockConfig {
                embed_dim,
                num_heads: config.block.num_heads,
                mlp_ratio: config.block.mlp_ratio,
                attn_dropout: config.block.attn_dropout,
                mlp_dropout: config.block.mlp_dropout,
                drop_path_rate: drop_path,
                activation: config.block.activation,
                layer_norm_eps: config.block.layer_norm_eps,
                qkv_bias: config.block.qkv_bias,
            };
            blocks.push(VitBlock::new(
                &block_cfg,
                rng.next_usize(u32::MAX as usize) as u64,
            )?);
        }

        let head = ClsHead::new(
            embed_dim,
            config.num_classes,
            rng.next_usize(u32::MAX as usize) as u64,
        )?;

        let mut state = VitState::new();
        state.register_layer("patch_embed", patch_embed.num_params());
        state.register_layer("cls_token", embed_dim);
        state.register_layer("pos_embed", pos_embed.embed.len());
        for (i, b) in blocks.iter().enumerate() {
            state.register_layer(&format!("block_{}", i), b.num_params());
        }
        state.register_layer("head", head.num_params());

        Ok(Self {
            config,
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            head,
            state,
        })
    }

    /// Full forward pass → [`VitOutput`].
    ///
    /// # Arguments
    /// - `pixels`: `[B, C, H, W]` flat image data.
    /// - `batch`: batch size.
    pub fn forward(&mut self, pixels: &[f64], batch: usize) -> VitResult<VitOutput> {
        if batch == 0 {
            return Err(VitError::EmptyBatch);
        }
        let embed_dim = self.config.embed_dim();
        let num_patches = self.config.num_patches();
        let seq_len = self.config.seq_len();

        // 1. Patch embedding
        let mut tokens = self.patch_embed.forward(pixels, batch)?;
        // tokens: [B, N, D]

        // 2. Prepend CLS token
        if self.config.use_cls_token {
            tokens =
                crate::ops::add_cls_token(&tokens, &self.cls_token, batch, num_patches, embed_dim)?;
        }

        // 3. Add position embedding
        self.pos_embed.add_to(&mut tokens, batch)?;

        // 4. Transformer blocks
        let mut tokens_3d = Tensor3D::from_data(batch, seq_len, embed_dim, tokens)?;
        for block in &self.blocks {
            tokens_3d = block.forward(&tokens_3d)?;
        }

        // 5. Pool
        let pooled = if self.config.use_cls_token {
            tokens_3d.cls_pool()
        } else {
            tokens_3d.mean_pool()
        };

        // 6. Head
        let logits_flat = self.head.forward(&pooled.data, batch)?;
        let logits: Vec<Vec<f64>> = (0..batch)
            .map(|b| {
                logits_flat[b * self.config.num_classes..(b + 1) * self.config.num_classes].to_vec()
            })
            .collect();

        let cls_token: Vec<Vec<f64>> = (0..batch)
            .map(|b| tokens_3d.batch_slice(b).data[..embed_dim].to_vec())
            .collect();
        let start = if self.config.use_cls_token { 1 } else { 0 };
        let patch_tokens: Vec<Vec<Vec<f64>>> = (0..batch)
            .map(|b| {
                (start..seq_len)
                    .map(|s| {
                        tokens_3d.data[b * seq_len * embed_dim + s * embed_dim
                            ..b * seq_len * embed_dim + (s + 1) * embed_dim]
                            .to_vec()
                    })
                    .collect()
            })
            .collect();

        self.state.record_forward((batch * seq_len) as u64);
        self.state.step();

        Ok(VitOutput {
            logits,
            cls_token,
            patch_tokens,
            attentions: vec![],
            feature_maps: HashMap::new(),
            reconstruction: None,
        })
    }

    /// Forward features only (no head).
    pub fn forward_features(&self, pixels: &[f64], batch: usize) -> VitResult<Tensor3D> {
        if batch == 0 {
            return Err(VitError::EmptyBatch);
        }
        let embed_dim = self.config.embed_dim();
        let num_patches = self.config.num_patches();
        let seq_len = self.config.seq_len();

        let mut tokens = self.patch_embed.forward(pixels, batch)?;
        if self.config.use_cls_token {
            tokens =
                crate::ops::add_cls_token(&tokens, &self.cls_token, batch, num_patches, embed_dim)?;
        }
        let mut pos_tokens = tokens.clone();
        self.pos_embed.add_to(&mut pos_tokens, batch)?;

        let mut tokens_3d = Tensor3D::from_data(batch, seq_len, embed_dim, pos_tokens)?;
        for block in &self.blocks {
            tokens_3d = block.forward(&tokens_3d)?;
        }
        Ok(tokens_3d)
    }

    /// Total trainable parameter count.
    pub fn total_params(&self) -> usize {
        self.state.total_params()
    }

    /// Set train/eval mode on all blocks.
    pub fn train(&mut self) {
        self.state.set_training(true);
        for b in &mut self.blocks {
            b.training = true;
        }
    }

    /// Set eval mode.
    pub fn eval(&mut self) {
        self.state.set_training(false);
        for b in &mut self.blocks {
            b.training = false;
        }
    }

    /// Model summary string.
    pub fn summary(&self) -> String {
        format!(
            "{}\nTotal params: {}",
            self.config.summary(),
            self.total_params()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VitConfig;

    fn micro_config() -> VitConfig {
        let mut cfg = VitConfig::default();
        cfg.patch_embed.image_size = 8;
        cfg.patch_embed.patch_size = 4;
        cfg.patch_embed.embed_dim = 16;
        cfg.patch_embed.in_channels = 1;
        cfg.block.embed_dim = 16;
        cfg.block.num_heads = 2;
        cfg.block.mlp_ratio = 2.0;
        cfg.depth = 2;
        cfg.num_classes = 4;
        cfg.pos_embed.seq_len = 5;
        cfg.pos_embed.embed_dim = 16;
        cfg.pos_embed.grid_h = 2;
        cfg.pos_embed.grid_w = 2;
        cfg
    }

    fn micro_pixels(batch: usize) -> Vec<f64> {
        vec![0.5f64; batch * 1 * 8 * 8]
    }

    #[test]
    fn test_vit_new() {
        let cfg = micro_config();
        let vit = ViT::new(cfg, 0).unwrap();
        assert_eq!(vit.blocks.len(), 2);
        assert_eq!(vit.config.embed_dim(), 16);
    }

    #[test]
    fn test_vit_forward_logits_shape() {
        let cfg = micro_config();
        let mut vit = ViT::new(cfg, 0).unwrap();
        let pixels = micro_pixels(3);
        let out = vit.forward(&pixels, 3).unwrap();
        assert_eq!(out.logits.len(), 3);
        assert_eq!(out.logits[0].len(), 4);
    }

    #[test]
    fn test_vit_forward_cls_shape() {
        let cfg = micro_config();
        let mut vit = ViT::new(cfg, 0).unwrap();
        let pixels = micro_pixels(2);
        let out = vit.forward(&pixels, 2).unwrap();
        assert_eq!(out.cls_token.len(), 2);
        assert_eq!(out.cls_token[0].len(), 16);
    }

    #[test]
    fn test_vit_forward_patch_tokens_shape() {
        let cfg = micro_config();
        let mut vit = ViT::new(cfg, 0).unwrap();
        let out = vit.forward(&micro_pixels(1), 1).unwrap();
        assert_eq!(out.patch_tokens[0].len(), 4); // 4 patches
    }

    #[test]
    fn test_vit_forward_empty_batch() {
        let cfg = micro_config();
        let mut vit = ViT::new(cfg, 0).unwrap();
        assert!(vit.forward(&[], 0).is_err());
    }

    #[test]
    fn test_vit_forward_features() {
        let cfg = micro_config();
        let vit = ViT::new(cfg, 0).unwrap();
        let feats = vit.forward_features(&micro_pixels(2), 2).unwrap();
        assert_eq!(feats.batch, 2);
        assert_eq!(feats.seq, 5); // 4 patches + CLS
    }

    #[test]
    fn test_vit_total_params() {
        let cfg = micro_config();
        let vit = ViT::new(cfg, 0).unwrap();
        assert!(vit.total_params() > 0);
    }

    #[test]
    fn test_vit_train_eval() {
        let cfg = micro_config();
        let mut vit = ViT::new(cfg, 0).unwrap();
        vit.eval();
        assert!(!vit.state.is_training);
        vit.train();
        assert!(vit.state.is_training);
    }

    #[test]
    fn test_vit_forward_finite() {
        let cfg = micro_config();
        let mut vit = ViT::new(cfg, 99).unwrap();
        let out = vit.forward(&micro_pixels(2), 2).unwrap();
        for row in &out.logits {
            for &v in row {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_vit_deterministic() {
        let cfg = micro_config();
        let mut v1 = ViT::new(cfg.clone(), 42).unwrap();
        let mut v2 = ViT::new(cfg, 42).unwrap();
        let p = micro_pixels(1);
        let o1 = v1.forward(&p, 1).unwrap();
        let o2 = v2.forward(&p, 1).unwrap();
        for (a, b) in o1.logits[0].iter().zip(o2.logits[0].iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_vit_invalid_config() {
        let mut cfg = micro_config();
        cfg.depth = 0;
        assert!(ViT::new(cfg, 0).is_err());
    }

    #[test]
    fn test_vit_state_updated() {
        let cfg = micro_config();
        let mut vit = ViT::new(cfg, 0).unwrap();
        vit.forward(&micro_pixels(1), 1).unwrap();
        assert_eq!(vit.state.forward_count, 1);
    }

    #[test]
    fn test_vit_summary_nonempty() {
        let cfg = micro_config();
        let vit = ViT::new(cfg, 0).unwrap();
        assert!(!vit.summary().is_empty());
    }

    #[test]
    fn test_vit_multiple_batches() {
        let cfg = micro_config();
        let mut vit = ViT::new(cfg, 0).unwrap();
        for batch in [1usize, 2, 4] {
            let out = vit.forward(&micro_pixels(batch), batch).unwrap();
            assert_eq!(out.logits.len(), batch);
        }
    }

    #[test]
    fn test_vit_blocks_count() {
        let cfg = micro_config();
        let vit = ViT::new(cfg, 0).unwrap();
        assert_eq!(vit.blocks.len(), 2);
    }

    #[test]
    fn test_vit_no_cls_gap_pool() {
        let mut cfg = micro_config();
        cfg.use_cls_token = false;
        cfg.pos_embed.seq_len = 4; // 4 patches, no CLS
        cfg.pos_embed.has_cls_token = false;
        let mut vit = ViT::new(cfg, 0).unwrap();
        let out = vit.forward(&micro_pixels(1), 1).unwrap();
        assert_eq!(out.logits.len(), 1);
    }

    #[test]
    fn test_vit_forward_features_dim() {
        let cfg = micro_config();
        let vit = ViT::new(cfg, 0).unwrap();
        let feats = vit.forward_features(&micro_pixels(1), 1).unwrap();
        assert_eq!(feats.dim, 16);
    }
}
