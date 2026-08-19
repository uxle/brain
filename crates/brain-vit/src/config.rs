//! # Configuration for brain-vit
//!
//! Central configuration types for the Vision Transformer crate.
//! [`VitConfig`] is the master configuration aggregating patch, position
//! embedding, encoder, head, and training settings.
//!
//! ## Design
//! - Validated at construction via [`VitConfig::validate`]
//! - Serializable to/from key-value strings for checkpointing
//! - Preset constructors for ViT-Tiny, Small, Base, Large

use std::collections::HashMap;
use std::fmt;
use crate::core::VitError;

/// Result type for configuration operations.
pub type CfgResult<T> = Result<T, VitError>;

/// Position embedding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PosEmbedType {
    /// Learned 1D absolute position embedding.
    #[default]
    Learned1D,
    /// Learned 2D grid position embedding.
    Learned2D,
    /// Fixed sinusoidal position embedding.
    Sinusoidal,
    /// No position embedding.
    None,
}

impl fmt::Display for PosEmbedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PosEmbedType::Learned1D => write!(f, "learned_1d"),
            PosEmbedType::Learned2D => write!(f, "learned_2d"),
            PosEmbedType::Sinusoidal => write!(f, "sinusoidal"),
            PosEmbedType::None => write!(f, "none"),
        }
    }
}

impl PosEmbedType {
    /// Parse from string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> CfgResult<Self> {
        match s.to_lowercase().as_str() {
            "learned_1d" | "learned" => Ok(PosEmbedType::Learned1D),
            "learned_2d" | "2d" => Ok(PosEmbedType::Learned2D),
            "sinusoidal" | "sin" => Ok(PosEmbedType::Sinusoidal),
            "none" => Ok(PosEmbedType::None),
            other => Err(VitError::Config(format!("Unknown pos_embed_type: {}", other))),
        }
    }
}

/// Head type for the ViT output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeadType {
    /// Standard linear classification head.
    #[default]
    Classification,
    /// Self-supervised projection head (DINO-lite).
    Projection,
    /// Pixel-level segmentation head.
    Segmentation,
    /// Detection feature map head.
    Detection,
    /// No head (backbone mode).
    NoHead,
}

impl fmt::Display for HeadType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeadType::Classification => write!(f, "classification"),
            HeadType::Projection => write!(f, "projection"),
            HeadType::Segmentation => write!(f, "segmentation"),
            HeadType::Detection => write!(f, "detection"),
            HeadType::NoHead => write!(f, "no_head"),
        }
    }
}

/// Activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// Gaussian Error Linear Unit.
    #[default]
    Gelu,
    /// Rectified Linear Unit.
    Relu,
    /// Sigmoid Linear Unit.
    Silu,
    /// No activation.
    Identity,
}

impl Activation {
    /// Apply the activation function to a scalar.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Gelu => {
                // Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let pi = std::f64::consts::PI;
                let inner = (2.0 / pi).sqrt() * (x + 0.044715 * x.powi(3));
                x * 0.5 * (1.0 + inner.tanh())
            }
            Activation::Relu => x.max(0.0),
            Activation::Silu => x / (1.0 + (-x).exp()),
            Activation::Identity => x,
        }
    }

    /// Apply to a slice, in-place.
    pub fn apply_slice(&self, xs: &mut [f64]) {
        for x in xs.iter_mut() {
            *x = self.apply(*x);
        }
    }
}

/// Patch embedding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PatchMode {
    /// Standard convolutional patch embedding.
    #[default]
    Conv,
    /// Unfold-based patch embedding (explicit reshape).
    Unfold,
}

/// Configuration for patch embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchEmbedConfig {
    /// Input image size (height == width).
    pub image_size: usize,
    /// Patch size (height == width).
    pub patch_size: usize,
    /// Number of input channels (e.g., 3 for RGB).
    pub in_channels: usize,
    /// Output embedding dimension.
    pub embed_dim: usize,
    /// Whether to use a projection bias.
    pub bias: bool,
    /// Patch embedding mode.
    pub mode: PatchMode,
}

impl Default for PatchEmbedConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            in_channels: 3,
            embed_dim: 768,
            bias: true,
            mode: PatchMode::Conv,
        }
    }
}

impl PatchEmbedConfig {
    /// Number of patches along one axis.
    pub fn grid_size(&self) -> usize { self.image_size / self.patch_size }

    /// Total number of patches.
    pub fn num_patches(&self) -> usize { self.grid_size() * self.grid_size() }

    /// Validate patch embedding config.
    pub fn validate(&self) -> CfgResult<()> {
        if self.patch_size == 0 {
            return Err(VitError::Config("patch_size must be > 0".to_string()));
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(VitError::InvalidPatchSize {
                image_dim: self.image_size,
                patch_size: self.patch_size,
            });
        }
        if self.embed_dim == 0 {
            return Err(VitError::Config("embed_dim must be > 0".to_string()));
        }
        if self.in_channels == 0 {
            return Err(VitError::Config("in_channels must be > 0".to_string()));
        }
        Ok(())
    }
}

/// Configuration for position embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct PosEmbedConfig {
    /// Total sequence length including CLS token.
    pub seq_len: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Type of position embedding.
    pub embed_type: PosEmbedType,
    /// Whether the CLS token is prepended.
    pub has_cls_token: bool,
    /// Grid height for 2D embeddings.
    pub grid_h: usize,
    /// Grid width for 2D embeddings.
    pub grid_w: usize,
    /// Dropout on position embeddings.
    pub dropout: f64,
}

impl Default for PosEmbedConfig {
    fn default() -> Self {
        Self {
            seq_len: 197,   // 196 patches + 1 CLS
            embed_dim: 768,
            embed_type: PosEmbedType::Learned1D,
            has_cls_token: true,
            grid_h: 14,
            grid_w: 14,
            dropout: 0.0,
        }
    }
}

impl PosEmbedConfig {
    /// Validate position embedding config.
    pub fn validate(&self) -> CfgResult<()> {
        if self.embed_dim == 0 {
            return Err(VitError::Config("pos_embed: embed_dim must be > 0".to_string()));
        }
        if self.seq_len == 0 {
            return Err(VitError::Config("pos_embed: seq_len must be > 0".to_string()));
        }
        Ok(())
    }
}

/// Configuration for a single ViT block.
#[derive(Debug, Clone, PartialEq)]
pub struct VitBlockConfig {
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// MLP hidden dimension ratio relative to embed_dim.
    pub mlp_ratio: f64,
    /// Attention dropout probability.
    pub attn_dropout: f64,
    /// MLP dropout probability.
    pub mlp_dropout: f64,
    /// Stochastic depth drop rate.
    pub drop_path_rate: f64,
    /// Activation function for MLP.
    pub activation: Activation,
    /// Layer norm epsilon.
    pub layer_norm_eps: f64,
    /// Whether to use bias in QKV projection.
    pub qkv_bias: bool,
}

impl Default for VitBlockConfig {
    fn default() -> Self {
        Self {
            embed_dim: 768,
            num_heads: 12,
            mlp_ratio: 4.0,
            attn_dropout: 0.0,
            mlp_dropout: 0.0,
            drop_path_rate: 0.0,
            activation: Activation::Gelu,
            layer_norm_eps: 1e-6,
            qkv_bias: true,
        }
    }
}

impl VitBlockConfig {
    /// MLP hidden dimension.
    pub fn mlp_dim(&self) -> usize {
        (self.embed_dim as f64 * self.mlp_ratio) as usize
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads.max(1)
    }

    /// Validate block config.
    pub fn validate(&self) -> CfgResult<()> {
        if self.embed_dim == 0 {
            return Err(VitError::Config("block: embed_dim must be > 0".to_string()));
        }
        if self.num_heads == 0 {
            return Err(VitError::Config("block: num_heads must be > 0".to_string()));
        }
        if !self.embed_dim.is_multiple_of(self.num_heads) {
            return Err(VitError::Config(format!(
                "block: embed_dim {} must be divisible by num_heads {}",
                self.embed_dim, self.num_heads
            )));
        }
        Ok(())
    }
}

/// Master ViT configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct VitConfig {
    /// Patch embedding configuration.
    pub patch_embed: PatchEmbedConfig,
    /// Position embedding configuration.
    pub pos_embed: PosEmbedConfig,
    /// Transformer block configuration (applied to all blocks uniformly).
    pub block: VitBlockConfig,
    /// Number of transformer encoder blocks.
    pub depth: usize,
    /// Number of output classes (for classification head).
    pub num_classes: usize,
    /// Head type.
    pub head_type: HeadType,
    /// Whether to prepend a CLS token.
    pub use_cls_token: bool,
    /// Embedding dropout.
    pub embed_dropout: f64,
    /// Whether to return attention weights during forward.
    pub return_attentions: bool,
    /// Stochastic depth drop path rates per block (overrides block.drop_path_rate).
    pub drop_path_rates: Vec<f64>,
    /// Global average pooling (no CLS token) mode.
    pub global_pool: bool,
}

impl Default for VitConfig {
    fn default() -> Self {
        Self {
            patch_embed: PatchEmbedConfig::default(),
            pos_embed: PosEmbedConfig::default(),
            block: VitBlockConfig::default(),
            depth: 12,
            num_classes: 1000,
            head_type: HeadType::Classification,
            use_cls_token: true,
            embed_dropout: 0.0,
            return_attentions: false,
            drop_path_rates: vec![],
            global_pool: false,
        }
    }
}

impl VitConfig {
    /// Validate the complete ViT configuration.
    pub fn validate(&self) -> CfgResult<()> {
        self.patch_embed.validate()?;
        self.pos_embed.validate()?;
        self.block.validate()?;
        if self.depth == 0 {
            return Err(VitError::Config("depth must be > 0".to_string()));
        }
        Ok(())
    }

    /// Number of patch tokens.
    pub fn num_patches(&self) -> usize { self.patch_embed.num_patches() }

    /// Total sequence length (patches + optional CLS).
    pub fn seq_len(&self) -> usize {
        self.num_patches() + if self.use_cls_token { 1 } else { 0 }
    }

    /// Embedding dimension.
    pub fn embed_dim(&self) -> usize { self.patch_embed.embed_dim }

    /// Per-block drop path rate using linear scaling.
    pub fn block_drop_path_rate(&self, block_idx: usize) -> f64 {
        if !self.drop_path_rates.is_empty() {
            self.drop_path_rates.get(block_idx).copied().unwrap_or(0.0)
        } else {
            self.block.drop_path_rate * block_idx as f64 / self.depth.max(1) as f64
        }
    }

    /// Generate a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "ViT-D{}H{}d{} | img={}px patch={}px embed={} classes={} pool={}",
            self.depth,
            self.block.num_heads,
            self.patch_embed.embed_dim,
            self.patch_embed.image_size,
            self.patch_embed.patch_size,
            self.patch_embed.embed_dim,
            self.num_classes,
            if self.use_cls_token { "cls" } else { "gap" },
        )
    }

    /// Serialize to key-value map.
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("image_size".to_string(), self.patch_embed.image_size.to_string());
        m.insert("patch_size".to_string(), self.patch_embed.patch_size.to_string());
        m.insert("embed_dim".to_string(), self.patch_embed.embed_dim.to_string());
        m.insert("depth".to_string(), self.depth.to_string());
        m.insert("num_heads".to_string(), self.block.num_heads.to_string());
        m.insert("mlp_ratio".to_string(), self.block.mlp_ratio.to_string());
        m.insert("num_classes".to_string(), self.num_classes.to_string());
        m.insert("use_cls_token".to_string(), self.use_cls_token.to_string());
        m
    }

    // ── Preset constructors ────────────────────────────────────────────────────

    /// ViT-Tiny: D=12 H=3 d=192.
    pub fn tiny() -> Self {
        let mut cfg = Self::default();
        cfg.patch_embed.embed_dim = 192;
        cfg.block.embed_dim = 192;
        cfg.block.num_heads = 3;
        cfg.depth = 12;
        cfg
    }

    /// ViT-Small: D=12 H=6 d=384.
    pub fn small() -> Self {
        let mut cfg = Self::default();
        cfg.patch_embed.embed_dim = 384;
        cfg.block.embed_dim = 384;
        cfg.block.num_heads = 6;
        cfg.depth = 12;
        cfg
    }

    /// ViT-Base: D=12 H=12 d=768.
    pub fn base() -> Self { Self::default() }

    /// ViT-Large: D=24 H=16 d=1024.
    pub fn large() -> Self {
        let mut cfg = Self::default();
        cfg.patch_embed.embed_dim = 1024;
        cfg.block.embed_dim = 1024;
        cfg.block.num_heads = 16;
        cfg.depth = 24;
        cfg
    }

    /// ViT-Huge: D=32 H=16 d=1280.
    pub fn huge() -> Self {
        let mut cfg = Self::default();
        cfg.patch_embed.embed_dim = 1280;
        cfg.block.embed_dim = 1280;
        cfg.block.num_heads = 16;
        cfg.depth = 32;
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pos_embed_type_from_str() {
        assert_eq!(PosEmbedType::from_str("learned_1d").unwrap(), PosEmbedType::Learned1D);
        assert_eq!(PosEmbedType::from_str("sin").unwrap(), PosEmbedType::Sinusoidal);
        assert_eq!(PosEmbedType::from_str("none").unwrap(), PosEmbedType::None);
        assert!(PosEmbedType::from_str("xyz").is_err());
    }

    #[test]
    fn test_activation_gelu() {
        let act = Activation::Gelu;
        // GELU(0) should be ~0
        assert!(act.apply(0.0).abs() < 0.01);
        // GELU(large positive) ≈ x
        let v = act.apply(10.0);
        assert!((v - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_activation_relu() {
        let act = Activation::Relu;
        assert_eq!(act.apply(-5.0), 0.0);
        assert!((act.apply(3.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_activation_silu() {
        let act = Activation::Silu;
        let v = act.apply(1.0);
        assert!(v > 0.5 && v < 1.0);
    }

    #[test]
    fn test_activation_identity() {
        let act = Activation::Identity;
        assert!((act.apply(3.14) - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_activation_apply_slice() {
        let act = Activation::Relu;
        let mut v = vec![-1.0, 0.0, 2.0, -3.0];
        act.apply_slice(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_patch_embed_config_default() {
        let cfg = PatchEmbedConfig::default();
        assert_eq!(cfg.grid_size(), 14);
        assert_eq!(cfg.num_patches(), 196);
    }

    #[test]
    fn test_patch_embed_config_validate_ok() {
        let cfg = PatchEmbedConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_patch_embed_config_validate_bad_patch() {
        let mut cfg = PatchEmbedConfig::default();
        cfg.patch_size = 15; // 224 % 15 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_patch_embed_config_validate_zero_embed() {
        let mut cfg = PatchEmbedConfig::default();
        cfg.embed_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_vit_block_config_mlp_dim() {
        let cfg = VitBlockConfig::default();
        assert_eq!(cfg.mlp_dim(), 768 * 4);
    }

    #[test]
    fn test_vit_block_config_head_dim() {
        let cfg = VitBlockConfig::default();
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn test_vit_block_config_validate_bad_heads() {
        let mut cfg = VitBlockConfig::default();
        cfg.num_heads = 7; // 768 % 7 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_vit_config_default_validate() {
        let cfg = VitConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_vit_config_num_patches() {
        let cfg = VitConfig::default();
        assert_eq!(cfg.num_patches(), 196);
    }

    #[test]
    fn test_vit_config_seq_len() {
        let cfg = VitConfig::default();
        assert_eq!(cfg.seq_len(), 197); // 196 + CLS
    }

    #[test]
    fn test_vit_config_no_cls_seq_len() {
        let mut cfg = VitConfig::default();
        cfg.use_cls_token = false;
        assert_eq!(cfg.seq_len(), 196);
    }

    #[test]
    fn test_vit_config_tiny() {
        let cfg = VitConfig::tiny();
        assert_eq!(cfg.embed_dim(), 192);
        assert_eq!(cfg.block.num_heads, 3);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_vit_config_small() {
        let cfg = VitConfig::small();
        assert_eq!(cfg.embed_dim(), 384);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_vit_config_large() {
        let cfg = VitConfig::large();
        assert_eq!(cfg.embed_dim(), 1024);
        assert_eq!(cfg.depth, 24);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_vit_config_summary() {
        let cfg = VitConfig::default();
        let s = cfg.summary();
        assert!(s.contains("ViT-D12"));
        assert!(s.contains("768"));
    }

    #[test]
    fn test_vit_config_to_map() {
        let cfg = VitConfig::default();
        let m = cfg.to_map();
        assert_eq!(m.get("depth").unwrap(), "12");
        assert_eq!(m.get("embed_dim").unwrap(), "768");
    }

    #[test]
    fn test_vit_config_drop_path_rate() {
        let mut cfg = VitConfig::default();
        cfg.block.drop_path_rate = 0.1;
        let r = cfg.block_drop_path_rate(6);
        assert!(r >= 0.0 && r <= 0.1);
    }

    #[test]
    fn test_vit_config_drop_path_rates_explicit() {
        let mut cfg = VitConfig::default();
        cfg.drop_path_rates = (0..12).map(|i| i as f64 * 0.01).collect();
        assert!((cfg.block_drop_path_rate(5) - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_pos_embed_config_validate() {
        let cfg = PosEmbedConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_pos_embed_config_validate_zero_dim() {
        let mut cfg = PosEmbedConfig::default();
        cfg.embed_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_vit_config_validate_zero_depth() {
        let mut cfg = VitConfig::default();
        cfg.depth = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_head_type_display() {
        assert_eq!(HeadType::Classification.to_string(), "classification");
        assert_eq!(HeadType::NoHead.to_string(), "no_head");
    }

    #[test]
    fn test_patch_mode_default() {
        assert_eq!(PatchMode::default(), PatchMode::Conv);
    }

    #[test]
    fn test_vit_config_huge() {
        let cfg = VitConfig::huge();
        assert_eq!(cfg.embed_dim(), 1280);
        assert_eq!(cfg.depth, 32);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_pos_embed_type_display() {
        assert_eq!(PosEmbedType::Sinusoidal.to_string(), "sinusoidal");
        assert_eq!(PosEmbedType::Learned2D.to_string(), "learned_2d");
    }

    #[test]
    fn test_vit_block_config_zero_heads_err() {
        let mut cfg = VitBlockConfig::default();
        cfg.num_heads = 0;
        assert!(cfg.validate().is_err());
    }
}
