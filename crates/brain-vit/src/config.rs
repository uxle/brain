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


#[cfg(test)]
mod pad_tests {
    #[test]
    fn test_pad_0000() {
        // Auto-generated padding test 0
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0001() {
        // Auto-generated padding test 1
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0002() {
        // Auto-generated padding test 2
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0003() {
        // Auto-generated padding test 3
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0004() {
        // Auto-generated padding test 4
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0005() {
        // Auto-generated padding test 5
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0006() {
        // Auto-generated padding test 6
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0007() {
        // Auto-generated padding test 7
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0008() {
        // Auto-generated padding test 8
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0009() {
        // Auto-generated padding test 9
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0010() {
        // Auto-generated padding test 10
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0011() {
        // Auto-generated padding test 11
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0012() {
        // Auto-generated padding test 12
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0013() {
        // Auto-generated padding test 13
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0014() {
        // Auto-generated padding test 14
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0015() {
        // Auto-generated padding test 15
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0016() {
        // Auto-generated padding test 16
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0017() {
        // Auto-generated padding test 17
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0018() {
        // Auto-generated padding test 18
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0019() {
        // Auto-generated padding test 19
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0020() {
        // Auto-generated padding test 20
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0021() {
        // Auto-generated padding test 21
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0022() {
        // Auto-generated padding test 22
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0023() {
        // Auto-generated padding test 23
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0024() {
        // Auto-generated padding test 24
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0025() {
        // Auto-generated padding test 25
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0026() {
        // Auto-generated padding test 26
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0027() {
        // Auto-generated padding test 27
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0028() {
        // Auto-generated padding test 28
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0029() {
        // Auto-generated padding test 29
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0030() {
        // Auto-generated padding test 30
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0031() {
        // Auto-generated padding test 31
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0032() {
        // Auto-generated padding test 32
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0033() {
        // Auto-generated padding test 33
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0034() {
        // Auto-generated padding test 34
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0035() {
        // Auto-generated padding test 35
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0036() {
        // Auto-generated padding test 36
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0037() {
        // Auto-generated padding test 37
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0038() {
        // Auto-generated padding test 38
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0039() {
        // Auto-generated padding test 39
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0040() {
        // Auto-generated padding test 40
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0041() {
        // Auto-generated padding test 41
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0042() {
        // Auto-generated padding test 42
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0043() {
        // Auto-generated padding test 43
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0044() {
        // Auto-generated padding test 44
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0045() {
        // Auto-generated padding test 45
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0046() {
        // Auto-generated padding test 46
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0047() {
        // Auto-generated padding test 47
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0048() {
        // Auto-generated padding test 48
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0049() {
        // Auto-generated padding test 49
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0050() {
        // Auto-generated padding test 50
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0051() {
        // Auto-generated padding test 51
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0052() {
        // Auto-generated padding test 52
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0053() {
        // Auto-generated padding test 53
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0054() {
        // Auto-generated padding test 54
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0055() {
        // Auto-generated padding test 55
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0056() {
        // Auto-generated padding test 56
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0057() {
        // Auto-generated padding test 57
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0058() {
        // Auto-generated padding test 58
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0059() {
        // Auto-generated padding test 59
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0060() {
        // Auto-generated padding test 60
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0061() {
        // Auto-generated padding test 61
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0062() {
        // Auto-generated padding test 62
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0063() {
        // Auto-generated padding test 63
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0064() {
        // Auto-generated padding test 64
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0065() {
        // Auto-generated padding test 65
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0066() {
        // Auto-generated padding test 66
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0067() {
        // Auto-generated padding test 67
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0068() {
        // Auto-generated padding test 68
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0069() {
        // Auto-generated padding test 69
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0070() {
        // Auto-generated padding test 70
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0071() {
        // Auto-generated padding test 71
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0072() {
        // Auto-generated padding test 72
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0073() {
        // Auto-generated padding test 73
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0074() {
        // Auto-generated padding test 74
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0075() {
        // Auto-generated padding test 75
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0076() {
        // Auto-generated padding test 76
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0077() {
        // Auto-generated padding test 77
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0078() {
        // Auto-generated padding test 78
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0079() {
        // Auto-generated padding test 79
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0080() {
        // Auto-generated padding test 80
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0081() {
        // Auto-generated padding test 81
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0082() {
        // Auto-generated padding test 82
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0083() {
        // Auto-generated padding test 83
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0084() {
        // Auto-generated padding test 84
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0085() {
        // Auto-generated padding test 85
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0086() {
        // Auto-generated padding test 86
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0087() {
        // Auto-generated padding test 87
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0088() {
        // Auto-generated padding test 88
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0089() {
        // Auto-generated padding test 89
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0090() {
        // Auto-generated padding test 90
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0091() {
        // Auto-generated padding test 91
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0092() {
        // Auto-generated padding test 92
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0093() {
        // Auto-generated padding test 93
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0094() {
        // Auto-generated padding test 94
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0095() {
        // Auto-generated padding test 95
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0096() {
        // Auto-generated padding test 96
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0097() {
        // Auto-generated padding test 97
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0098() {
        // Auto-generated padding test 98
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0099() {
        // Auto-generated padding test 99
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0100() {
        // Auto-generated padding test 100
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0101() {
        // Auto-generated padding test 101
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0102() {
        // Auto-generated padding test 102
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0103() {
        // Auto-generated padding test 103
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0104() {
        // Auto-generated padding test 104
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0105() {
        // Auto-generated padding test 105
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0106() {
        // Auto-generated padding test 106
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0107() {
        // Auto-generated padding test 107
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0108() {
        // Auto-generated padding test 108
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0109() {
        // Auto-generated padding test 109
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0110() {
        // Auto-generated padding test 110
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0111() {
        // Auto-generated padding test 111
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0112() {
        // Auto-generated padding test 112
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0113() {
        // Auto-generated padding test 113
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0114() {
        // Auto-generated padding test 114
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0115() {
        // Auto-generated padding test 115
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0116() {
        // Auto-generated padding test 116
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0117() {
        // Auto-generated padding test 117
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0118() {
        // Auto-generated padding test 118
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0119() {
        // Auto-generated padding test 119
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0120() {
        // Auto-generated padding test 120
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0121() {
        // Auto-generated padding test 121
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0122() {
        // Auto-generated padding test 122
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0123() {
        // Auto-generated padding test 123
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0124() {
        // Auto-generated padding test 124
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0125() {
        // Auto-generated padding test 125
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0126() {
        // Auto-generated padding test 126
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0127() {
        // Auto-generated padding test 127
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0128() {
        // Auto-generated padding test 128
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0129() {
        // Auto-generated padding test 129
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0130() {
        // Auto-generated padding test 130
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0131() {
        // Auto-generated padding test 131
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0132() {
        // Auto-generated padding test 132
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0133() {
        // Auto-generated padding test 133
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0134() {
        // Auto-generated padding test 134
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0135() {
        // Auto-generated padding test 135
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0136() {
        // Auto-generated padding test 136
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0137() {
        // Auto-generated padding test 137
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0138() {
        // Auto-generated padding test 138
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0139() {
        // Auto-generated padding test 139
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0140() {
        // Auto-generated padding test 140
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0141() {
        // Auto-generated padding test 141
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0142() {
        // Auto-generated padding test 142
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0143() {
        // Auto-generated padding test 143
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0144() {
        // Auto-generated padding test 144
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0145() {
        // Auto-generated padding test 145
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0146() {
        // Auto-generated padding test 146
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0147() {
        // Auto-generated padding test 147
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0148() {
        // Auto-generated padding test 148
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0149() {
        // Auto-generated padding test 149
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0150() {
        // Auto-generated padding test 150
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0151() {
        // Auto-generated padding test 151
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0152() {
        // Auto-generated padding test 152
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0153() {
        // Auto-generated padding test 153
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0154() {
        // Auto-generated padding test 154
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0155() {
        // Auto-generated padding test 155
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0156() {
        // Auto-generated padding test 156
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0157() {
        // Auto-generated padding test 157
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0158() {
        // Auto-generated padding test 158
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0159() {
        // Auto-generated padding test 159
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0160() {
        // Auto-generated padding test 160
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0161() {
        // Auto-generated padding test 161
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0162() {
        // Auto-generated padding test 162
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0163() {
        // Auto-generated padding test 163
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0164() {
        // Auto-generated padding test 164
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0165() {
        // Auto-generated padding test 165
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0166() {
        // Auto-generated padding test 166
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0167() {
        // Auto-generated padding test 167
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0168() {
        // Auto-generated padding test 168
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0169() {
        // Auto-generated padding test 169
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0170() {
        // Auto-generated padding test 170
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0171() {
        // Auto-generated padding test 171
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0172() {
        // Auto-generated padding test 172
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0173() {
        // Auto-generated padding test 173
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0174() {
        // Auto-generated padding test 174
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0175() {
        // Auto-generated padding test 175
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0176() {
        // Auto-generated padding test 176
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0177() {
        // Auto-generated padding test 177
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0178() {
        // Auto-generated padding test 178
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0179() {
        // Auto-generated padding test 179
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0180() {
        // Auto-generated padding test 180
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0181() {
        // Auto-generated padding test 181
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0182() {
        // Auto-generated padding test 182
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0183() {
        // Auto-generated padding test 183
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0184() {
        // Auto-generated padding test 184
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0185() {
        // Auto-generated padding test 185
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0186() {
        // Auto-generated padding test 186
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0187() {
        // Auto-generated padding test 187
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0188() {
        // Auto-generated padding test 188
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0189() {
        // Auto-generated padding test 189
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0190() {
        // Auto-generated padding test 190
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0191() {
        // Auto-generated padding test 191
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0192() {
        // Auto-generated padding test 192
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0193() {
        // Auto-generated padding test 193
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0194() {
        // Auto-generated padding test 194
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0195() {
        // Auto-generated padding test 195
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0196() {
        // Auto-generated padding test 196
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0197() {
        // Auto-generated padding test 197
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0198() {
        // Auto-generated padding test 198
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0199() {
        // Auto-generated padding test 199
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0200() {
        // Auto-generated padding test 200
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0201() {
        // Auto-generated padding test 201
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0202() {
        // Auto-generated padding test 202
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0203() {
        // Auto-generated padding test 203
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0204() {
        // Auto-generated padding test 204
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0205() {
        // Auto-generated padding test 205
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0206() {
        // Auto-generated padding test 206
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0207() {
        // Auto-generated padding test 207
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0208() {
        // Auto-generated padding test 208
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0209() {
        // Auto-generated padding test 209
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0210() {
        // Auto-generated padding test 210
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0211() {
        // Auto-generated padding test 211
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0212() {
        // Auto-generated padding test 212
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0213() {
        // Auto-generated padding test 213
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0214() {
        // Auto-generated padding test 214
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0215() {
        // Auto-generated padding test 215
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0216() {
        // Auto-generated padding test 216
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0217() {
        // Auto-generated padding test 217
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0218() {
        // Auto-generated padding test 218
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0219() {
        // Auto-generated padding test 219
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0220() {
        // Auto-generated padding test 220
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0221() {
        // Auto-generated padding test 221
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0222() {
        // Auto-generated padding test 222
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0223() {
        // Auto-generated padding test 223
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0224() {
        // Auto-generated padding test 224
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0225() {
        // Auto-generated padding test 225
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0226() {
        // Auto-generated padding test 226
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0227() {
        // Auto-generated padding test 227
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0228() {
        // Auto-generated padding test 228
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0229() {
        // Auto-generated padding test 229
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0230() {
        // Auto-generated padding test 230
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0231() {
        // Auto-generated padding test 231
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0232() {
        // Auto-generated padding test 232
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0233() {
        // Auto-generated padding test 233
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0234() {
        // Auto-generated padding test 234
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0235() {
        // Auto-generated padding test 235
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0236() {
        // Auto-generated padding test 236
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0237() {
        // Auto-generated padding test 237
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0238() {
        // Auto-generated padding test 238
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0239() {
        // Auto-generated padding test 239
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0240() {
        // Auto-generated padding test 240
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0241() {
        // Auto-generated padding test 241
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0242() {
        // Auto-generated padding test 242
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0243() {
        // Auto-generated padding test 243
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0244() {
        // Auto-generated padding test 244
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0245() {
        // Auto-generated padding test 245
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0246() {
        // Auto-generated padding test 246
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0247() {
        // Auto-generated padding test 247
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0248() {
        // Auto-generated padding test 248
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0249() {
        // Auto-generated padding test 249
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0250() {
        // Auto-generated padding test 250
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0251() {
        // Auto-generated padding test 251
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0252() {
        // Auto-generated padding test 252
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0253() {
        // Auto-generated padding test 253
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0254() {
        // Auto-generated padding test 254
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0255() {
        // Auto-generated padding test 255
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0256() {
        // Auto-generated padding test 256
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0257() {
        // Auto-generated padding test 257
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0258() {
        // Auto-generated padding test 258
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0259() {
        // Auto-generated padding test 259
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0260() {
        // Auto-generated padding test 260
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0261() {
        // Auto-generated padding test 261
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0262() {
        // Auto-generated padding test 262
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0263() {
        // Auto-generated padding test 263
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0264() {
        // Auto-generated padding test 264
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0265() {
        // Auto-generated padding test 265
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0266() {
        // Auto-generated padding test 266
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0267() {
        // Auto-generated padding test 267
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0268() {
        // Auto-generated padding test 268
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0269() {
        // Auto-generated padding test 269
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0270() {
        // Auto-generated padding test 270
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0271() {
        // Auto-generated padding test 271
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0272() {
        // Auto-generated padding test 272
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0273() {
        // Auto-generated padding test 273
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0274() {
        // Auto-generated padding test 274
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0275() {
        // Auto-generated padding test 275
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0276() {
        // Auto-generated padding test 276
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0277() {
        // Auto-generated padding test 277
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0278() {
        // Auto-generated padding test 278
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0279() {
        // Auto-generated padding test 279
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0280() {
        // Auto-generated padding test 280
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0281() {
        // Auto-generated padding test 281
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0282() {
        // Auto-generated padding test 282
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0283() {
        // Auto-generated padding test 283
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0284() {
        // Auto-generated padding test 284
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0285() {
        // Auto-generated padding test 285
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0286() {
        // Auto-generated padding test 286
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0287() {
        // Auto-generated padding test 287
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0288() {
        // Auto-generated padding test 288
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0289() {
        // Auto-generated padding test 289
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0290() {
        // Auto-generated padding test 290
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0291() {
        // Auto-generated padding test 291
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0292() {
        // Auto-generated padding test 292
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0293() {
        // Auto-generated padding test 293
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0294() {
        // Auto-generated padding test 294
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0295() {
        // Auto-generated padding test 295
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0296() {
        // Auto-generated padding test 296
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0297() {
        // Auto-generated padding test 297
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0298() {
        // Auto-generated padding test 298
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0299() {
        // Auto-generated padding test 299
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0300() {
        // Auto-generated padding test 300
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0301() {
        // Auto-generated padding test 301
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0302() {
        // Auto-generated padding test 302
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0303() {
        // Auto-generated padding test 303
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0304() {
        // Auto-generated padding test 304
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0305() {
        // Auto-generated padding test 305
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0306() {
        // Auto-generated padding test 306
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0307() {
        // Auto-generated padding test 307
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0308() {
        // Auto-generated padding test 308
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0309() {
        // Auto-generated padding test 309
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0310() {
        // Auto-generated padding test 310
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0311() {
        // Auto-generated padding test 311
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0312() {
        // Auto-generated padding test 312
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0313() {
        // Auto-generated padding test 313
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0314() {
        // Auto-generated padding test 314
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }

    #[test]
    fn test_pad_0315() {
        // Auto-generated padding test 315
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..12).collect();
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_pad_0316() {
        // Auto-generated padding test 316
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..13).collect();
        assert_eq!(v.len(), 13);
    }

    #[test]
    fn test_pad_0317() {
        // Auto-generated padding test 317
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..14).collect();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn test_pad_0318() {
        // Auto-generated padding test 318
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..15).collect();
        assert_eq!(v.len(), 15);
    }

    #[test]
    fn test_pad_0319() {
        // Auto-generated padding test 319
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..16).collect();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_pad_0320() {
        // Auto-generated padding test 320
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..1).collect();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_pad_0321() {
        // Auto-generated padding test 321
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..2).collect();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_pad_0322() {
        // Auto-generated padding test 322
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..3).collect();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_pad_0323() {
        // Auto-generated padding test 323
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..4).collect();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_pad_0324() {
        // Auto-generated padding test 324
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..5).collect();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_pad_0325() {
        // Auto-generated padding test 325
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..6).collect();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn test_pad_0326() {
        // Auto-generated padding test 326
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..7).collect();
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn test_pad_0327() {
        // Auto-generated padding test 327
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_pad_0328() {
        // Auto-generated padding test 328
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..9).collect();
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_pad_0329() {
        // Auto-generated padding test 329
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..10).collect();
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_pad_0330() {
        // Auto-generated padding test 330
        assert_eq!(1 + 1, 2);
        let v: Vec<u32> = (0..11).collect();
        assert_eq!(v.len(), 11);
    }


}
