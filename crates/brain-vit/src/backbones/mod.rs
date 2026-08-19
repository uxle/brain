//! # ViT Backbone Variants for brain-vit
//!
//! Backbone configurations (Tiny/Small/Base/Large) exposing features
//! without a task-specific head, suitable for transfer learning.

use crate::core::{VitResult, Tensor3D};
use crate::config::VitConfig;
use crate::vit::ViT;

/// Named backbone size presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneSize {
    /// ViT-Tiny: D=12, H=3, d=192.
    Tiny,
    /// ViT-Small: D=12, H=6, d=384.
    Small,
    /// ViT-Base: D=12, H=12, d=768.
    Base,
    /// ViT-Large: D=24, H=16, d=1024.
    Large,
}

impl BackboneSize {
    /// Convert to a VitConfig.
    pub fn config(self) -> VitConfig {
        match self {
            BackboneSize::Tiny => VitConfig::tiny(),
            BackboneSize::Small => VitConfig::small(),
            BackboneSize::Base => VitConfig::base(),
            BackboneSize::Large => VitConfig::large(),
        }
    }

    /// Embedding dimension for this backbone.
    pub fn embed_dim(self) -> usize {
        match self {
            BackboneSize::Tiny => 192,
            BackboneSize::Small => 384,
            BackboneSize::Base => 768,
            BackboneSize::Large => 1024,
        }
    }

    /// Number of transformer blocks.
    pub fn depth(self) -> usize {
        match self {
            BackboneSize::Tiny => 12,
            BackboneSize::Small => 12,
            BackboneSize::Base => 12,
            BackboneSize::Large => 24,
        }
    }
}

/// Backbone configuration.
#[derive(Debug, Clone)]
pub struct BackboneConfig {
    /// Which preset to use.
    pub size: BackboneSize,
    /// Whether to include the CLS token.
    pub use_cls_token: bool,
    /// Whether to return intermediate features.
    pub return_intermediate: bool,
    /// Which block indices to return features from (empty = last only).
    pub return_blocks: Vec<usize>,
}

impl Default for BackboneConfig {
    fn default() -> Self {
        Self {
            size: BackboneSize::Base,
            use_cls_token: true,
            return_intermediate: false,
            return_blocks: vec![],
        }
    }
}

/// ViT backbone (feature extractor without classification head).
pub struct VitBackbone {
    /// Inner ViT (head is ignored).
    pub vit: ViT,
    /// Backbone config.
    pub backbone_config: BackboneConfig,
}

impl VitBackbone {
    /// Create backbone from a named size preset (with micro image for tests).
    ///
    /// # Example
    /// ```rust
    /// use brain_vit::backbones::{VitBackbone, BackboneSize};
    /// // Backbone creation is expensive for large models; use micro configs in tests.
    /// ```
    pub fn from_size(size: BackboneSize, seed: u64) -> VitResult<Self> {
        let vit_cfg = size.config();
        Self::new(vit_cfg, BackboneConfig { size, ..Default::default() }, seed)
    }

    /// Create backbone from a full ViT config.
    pub fn new(vit_cfg: VitConfig, backbone_config: BackboneConfig, seed: u64) -> VitResult<Self> {
        let vit = ViT::new(vit_cfg, seed)?;
        Ok(Self { vit, backbone_config })
    }

    /// Extract features from pixels.
    ///
    /// Returns `[B, seq_len, embed_dim]` token tensor.
    pub fn extract_features(&self, pixels: &[f64], batch: usize) -> VitResult<Tensor3D> {
        self.vit.forward_features(pixels, batch)
    }

    /// Extract CLS token only → `[B, embed_dim]` flat.
    pub fn extract_cls(&self, pixels: &[f64], batch: usize) -> VitResult<Vec<f64>> {
        let feats = self.extract_features(pixels, batch)?;
        let embed_dim = feats.dim;
        Ok((0..batch).flat_map(|b|
            feats.data[b * feats.seq * embed_dim..b * feats.seq * embed_dim + embed_dim].iter().copied()
        ).collect())
    }

    /// Extract patch tokens only → `[B, N, D]` flat.
    pub fn extract_patches(&self, pixels: &[f64], batch: usize) -> VitResult<Vec<f64>> {
        let feats = self.extract_features(pixels, batch)?;
        let d = feats.dim;
        let start = if self.vit.config.use_cls_token { 1 } else { 0 };
        let n_patches = feats.seq - start;
        let mut out = vec![0.0f64; batch * n_patches * d];
        for b in 0..batch {
            let src = b * feats.seq * d + start * d;
            let dst = b * n_patches * d;
            out[dst..dst + n_patches * d].copy_from_slice(
                &feats.data[src..src + n_patches * d]);
        }
        Ok(out)
    }

    /// Embedding dimension of this backbone.
    pub fn embed_dim(&self) -> usize { self.vit.config.embed_dim() }

    /// Total parameter count.
    pub fn total_params(&self) -> usize { self.vit.total_params() }
}

/// Create a micro backbone for testing purposes (tiny config, small image).
pub fn micro_backbone(seed: u64) -> VitResult<VitBackbone> {
    let mut cfg = VitConfig::default();
    cfg.patch_embed.image_size = 8;
    cfg.patch_embed.patch_size = 4;
    cfg.patch_embed.embed_dim = 16;
    cfg.patch_embed.in_channels = 1;
    cfg.block.embed_dim = 16;
    cfg.block.num_heads = 2;
    cfg.block.mlp_ratio = 2.0;
    cfg.depth = 1;
    cfg.num_classes = 4;
    cfg.pos_embed.seq_len = 5;
    cfg.pos_embed.embed_dim = 16;
    cfg.pos_embed.grid_h = 2;
    cfg.pos_embed.grid_w = 2;
    let backbone_cfg = BackboneConfig {
        size: BackboneSize::Tiny,
        use_cls_token: true,
        return_intermediate: false,
        return_blocks: vec![],
    };
    VitBackbone::new(cfg, backbone_cfg, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backbone_size_embed_dim() {
        assert_eq!(BackboneSize::Tiny.embed_dim(), 192);
        assert_eq!(BackboneSize::Base.embed_dim(), 768);
        assert_eq!(BackboneSize::Large.embed_dim(), 1024);
    }

    #[test]
    fn test_backbone_size_depth() {
        assert_eq!(BackboneSize::Tiny.depth(), 12);
        assert_eq!(BackboneSize::Large.depth(), 24);
    }

    #[test]
    fn test_micro_backbone_creation() {
        let bb = micro_backbone(0).unwrap();
        assert_eq!(bb.embed_dim(), 16);
    }

    #[test]
    fn test_micro_backbone_extract_features() {
        let bb = micro_backbone(0).unwrap();
        let pixels = vec![0.5f64; 2 * 1 * 8 * 8];
        let feats = bb.extract_features(&pixels, 2).unwrap();
        assert_eq!(feats.batch, 2);
        assert_eq!(feats.dim, 16);
    }

    #[test]
    fn test_micro_backbone_extract_cls() {
        let bb = micro_backbone(0).unwrap();
        let pixels = vec![0.5f64; 2 * 1 * 8 * 8];
        let cls = bb.extract_cls(&pixels, 2).unwrap();
        assert_eq!(cls.len(), 2 * 16);
    }

    #[test]
    fn test_micro_backbone_extract_patches() {
        let bb = micro_backbone(0).unwrap();
        let pixels = vec![0.5f64; 1 * 1 * 8 * 8];
        let patches = bb.extract_patches(&pixels, 1).unwrap();
        assert_eq!(patches.len(), 1 * 4 * 16); // 4 patches, 16 dim
    }

    #[test]
    fn test_micro_backbone_total_params() {
        let bb = micro_backbone(0).unwrap();
        assert!(bb.total_params() > 0);
    }

    #[test]
    fn test_backbone_config_default() {
        let cfg = BackboneConfig::default();
        assert_eq!(cfg.size, BackboneSize::Base);
        assert!(cfg.use_cls_token);
    }

    #[test]
    fn test_backbone_features_finite() {
        let bb = micro_backbone(0).unwrap();
        let pixels = vec![0.3f64; 1 * 1 * 8 * 8];
        let feats = bb.extract_features(&pixels, 1).unwrap();
        assert!(feats.data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_backbone_cls_finite() {
        let bb = micro_backbone(1).unwrap();
        let pixels = vec![0.5f64; 1 * 1 * 8 * 8];
        let cls = bb.extract_cls(&pixels, 1).unwrap();
        assert!(cls.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_backbone_size_config_valid() {
        for size in [BackboneSize::Tiny, BackboneSize::Small] {
            let cfg = size.config();
            assert!(cfg.validate().is_ok());
        }
    }

    #[test]
    fn test_backbone_extract_patches_shape() {
        let bb = micro_backbone(0).unwrap();
        let pixels = vec![0.1f64; 3 * 1 * 8 * 8];
        let patches = bb.extract_patches(&pixels, 3).unwrap();
        assert_eq!(patches.len(), 3 * 4 * 16);
    }

    #[test]
    fn test_backbone_empty_batch_err() {
        let bb = micro_backbone(0).unwrap();
        assert!(bb.extract_features(&[], 0).is_err());
    }

    #[test]
    fn test_backbone_different_seeds_different_params() {
        let b1 = micro_backbone(1).unwrap();
        let b2 = micro_backbone(2).unwrap();
        assert_ne!(b1.vit.cls_token, b2.vit.cls_token);
    }

    #[test]
    fn test_backbone_same_seed_same_params() {
        let b1 = micro_backbone(42).unwrap();
        let b2 = micro_backbone(42).unwrap();
        assert_eq!(b1.vit.cls_token, b2.vit.cls_token);
    }

    #[test]
    fn test_backbone_features_seq_len() {
        let bb = micro_backbone(0).unwrap();
        let pixels = vec![0.5f64; 1 * 1 * 8 * 8];
        let feats = bb.extract_features(&pixels, 1).unwrap();
        assert_eq!(feats.seq, 5); // 4 patches + CLS
    }
}
