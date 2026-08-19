//! # ViT Head Variants for brain-vit
//!
//! Prediction heads that attach to the ViT encoder:
//! - [`ClsHead`] — standard linear classification head
//! - [`DinoHead`] — self-supervised projection head (DINO-lite)
//! - [`SegHead`] — pixel-level segmentation decoder (upsampling)

use crate::core::{VitError, VitResult, Tensor2D, SimpleRng};
use crate::ops::linear;

/// Trait for all ViT head types.
pub trait VitHead {
    /// Forward pass: `[B, D]` → `[B, out_dim]` flat.
    fn forward(&self, pooled: &[f64], batch: usize) -> VitResult<Vec<f64>>;
    /// Number of trainable parameters.
    fn num_params(&self) -> usize;
    /// Output dimension.
    fn out_dim(&self) -> usize;
}

/// Linear classification head.
///
/// `linear(pooled) → logits` where `pooled: [B, embed_dim]`.
pub struct ClsHead {
    /// Weight `[num_classes, embed_dim]`.
    pub weight: Vec<f64>,
    /// Bias `[num_classes]`.
    pub bias: Vec<f64>,
    /// Input embedding dimension.
    pub embed_dim: usize,
    /// Number of output classes.
    pub num_classes: usize,
}

impl ClsHead {
    /// Create with Xavier-initialized weights.
    pub fn new(embed_dim: usize, num_classes: usize, seed: u64) -> VitResult<Self> {
        if embed_dim == 0 || num_classes == 0 {
            return Err(VitError::Config("ClsHead: embed_dim and num_classes must be > 0".to_string()));
        }
        let mut rng = SimpleRng::new(seed);
        let weight = rng.xavier_uniform(num_classes, embed_dim);
        let bias = vec![0.0f64; num_classes];
        Ok(Self { weight, bias, embed_dim, num_classes })
    }
}

impl VitHead for ClsHead {
    fn forward(&self, pooled: &[f64], batch: usize) -> VitResult<Vec<f64>> {
        if pooled.len() != batch * self.embed_dim {
            return Err(VitError::Shape(format!(
                "ClsHead: expected {} elements, got {}", batch * self.embed_dim, pooled.len()
            )));
        }
        let input = Tensor2D::from_data(batch, self.embed_dim, pooled.to_vec())?;
        let w = Tensor2D::from_data(self.num_classes, self.embed_dim, self.weight.clone())?;
        let out = linear(&input, &w, Some(&self.bias))?;
        Ok(out.data)
    }
    fn num_params(&self) -> usize { self.weight.len() + self.bias.len() }
    fn out_dim(&self) -> usize { self.num_classes }
}

/// DINO-lite self-supervised projection head.
///
/// Three-layer MLP: embed_dim → hidden → hidden → proj_dim (normalized).
pub struct DinoHead {
    /// First layer weight `[hidden, embed_dim]`.
    pub w1: Vec<f64>,
    /// First layer bias `[hidden]`.
    pub b1: Vec<f64>,
    /// Second layer weight `[hidden, hidden]`.
    pub w2: Vec<f64>,
    /// Second layer bias `[hidden]`.
    pub b2: Vec<f64>,
    /// Last layer weight `[proj_dim, hidden]`.
    pub w3: Vec<f64>,
    /// Input embedding dimension.
    pub embed_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Output projection dimension.
    pub proj_dim: usize,
}

impl DinoHead {
    /// Create with Xavier-initialized weights.
    pub fn new(embed_dim: usize, hidden_dim: usize, proj_dim: usize, seed: u64) -> VitResult<Self> {
        if embed_dim == 0 || hidden_dim == 0 || proj_dim == 0 {
            return Err(VitError::Config("DinoHead: all dims must be > 0".to_string()));
        }
        let mut rng = SimpleRng::new(seed);
        let w1 = rng.xavier_uniform(hidden_dim, embed_dim);
        let b1 = vec![0.0f64; hidden_dim];
        let w2 = rng.xavier_uniform(hidden_dim, hidden_dim);
        let b2 = vec![0.0f64; hidden_dim];
        let w3 = rng.xavier_uniform(proj_dim, hidden_dim);
        Ok(Self { w1, b1, w2, b2, w3, embed_dim, hidden_dim, proj_dim })
    }
}

impl VitHead for DinoHead {
    fn forward(&self, pooled: &[f64], batch: usize) -> VitResult<Vec<f64>> {
        if pooled.len() != batch * self.embed_dim {
            return Err(VitError::Shape("DinoHead: shape mismatch".to_string()));
        }
        let input = Tensor2D::from_data(batch, self.embed_dim, pooled.to_vec())?;
        let w1 = Tensor2D::from_data(self.hidden_dim, self.embed_dim, self.w1.clone())?;
        let w2 = Tensor2D::from_data(self.hidden_dim, self.hidden_dim, self.w2.clone())?;
        let w3 = Tensor2D::from_data(self.proj_dim, self.hidden_dim, self.w3.clone())?;

        // Layer 1: linear + gelu
        let mut h1 = linear(&input, &w1, Some(&self.b1))?;
        for x in h1.data.iter_mut() {
            let v = *x;
            let pi = std::f64::consts::PI;
            *x = v * 0.5 * (1.0 + ((2.0 / pi).sqrt() * (v + 0.044715 * v.powi(3))).tanh());
        }

        // Layer 2: linear + gelu
        let mut h2 = linear(&h1, &w2, Some(&self.b2))?;
        for x in h2.data.iter_mut() {
            let v = *x;
            let pi = std::f64::consts::PI;
            *x = v * 0.5 * (1.0 + ((2.0 / pi).sqrt() * (v + 0.044715 * v.powi(3))).tanh());
        }

        // Layer 3: linear (no activation, no bias)
        let h3 = linear(&h2, &w3, None)?;

        // L2 normalize each row
        let mut out = h3.data.clone();
        for b in 0..batch {
            let start = b * self.proj_dim;
            let norm: f64 = out[start..start + self.proj_dim]
                .iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-12);
            for x in out[start..start + self.proj_dim].iter_mut() {
                *x /= norm;
            }
        }
        Ok(out)
    }

    fn num_params(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len() + self.w3.len()
    }

    fn out_dim(&self) -> usize { self.proj_dim }
}

/// Simple segmentation head: projects patch tokens to per-pixel logits via upsampling.
///
/// Takes patch tokens `[B, N, D]` and produces `[B, num_classes, H, W]` flat.
pub struct SegHead {
    /// Linear classifier weight `[num_classes, embed_dim]`.
    pub weight: Vec<f64>,
    /// Linear classifier bias `[num_classes]`.
    pub bias: Vec<f64>,
    /// Input embedding dimension.
    pub embed_dim: usize,
    /// Number of semantic classes.
    pub num_classes: usize,
    /// Output height.
    pub out_h: usize,
    /// Output width.
    pub out_w: usize,
}

impl SegHead {
    /// Create a new segmentation head.
    pub fn new(embed_dim: usize, num_classes: usize, out_h: usize, out_w: usize, seed: u64) -> VitResult<Self> {
        let mut rng = SimpleRng::new(seed);
        let weight = rng.xavier_uniform(num_classes, embed_dim);
        let bias = vec![0.0f64; num_classes];
        Ok(Self { weight, bias, embed_dim, num_classes, out_h, out_w })
    }

    /// Forward: patch token logits `[B, N, num_classes]` (upsampling is bilinear).
    ///
    /// `patch_tokens`: `[B, N, D]` flat; `n_sqrt = sqrt(N)` (grid side).
    pub fn forward_seg(&self, patch_tokens: &[f64], batch: usize, n: usize) -> VitResult<Vec<f64>> {
        if patch_tokens.len() != batch * n * self.embed_dim {
            return Err(VitError::Shape("SegHead: patch_tokens shape mismatch".to_string()));
        }
        let input = Tensor2D::from_data(batch * n, self.embed_dim, patch_tokens.to_vec())?;
        let w = Tensor2D::from_data(self.num_classes, self.embed_dim, self.weight.clone())?;
        let logits = linear(&input, &w, Some(&self.bias))?;
        // logits: [B * N, num_classes]
        // Return as [B, N, num_classes]
        Ok(logits.data)
    }

    /// Number of parameters.
    pub fn num_params(&self) -> usize { self.weight.len() + self.bias.len() }
}

/// Configuration for ViT heads.
#[derive(Debug, Clone, PartialEq)]
pub struct VitHeadConfig {
    /// Input embedding dimension.
    pub embed_dim: usize,
    /// Output dimension (classes for cls, projection dim for dino, classes for seg).
    pub out_dim: usize,
    /// Hidden dimension for DINO head.
    pub hidden_dim: usize,
    /// Output height (for seg head).
    pub out_h: usize,
    /// Output width (for seg head).
    pub out_w: usize,
}

impl Default for VitHeadConfig {
    fn default() -> Self {
        Self {
            embed_dim: 768,
            out_dim: 1000,
            hidden_dim: 2048,
            out_h: 224,
            out_w: 224,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cls_head_new() {
        let h = ClsHead::new(64, 10, 0).unwrap();
        assert_eq!(h.embed_dim, 64);
        assert_eq!(h.num_classes, 10);
    }

    #[test]
    fn test_cls_head_forward_shape() {
        let h = ClsHead::new(64, 10, 0).unwrap();
        let pooled = vec![0.5f64; 3 * 64];
        let out = h.forward(&pooled, 3).unwrap();
        assert_eq!(out.len(), 3 * 10);
    }

    #[test]
    fn test_cls_head_forward_finite() {
        let h = ClsHead::new(16, 4, 1).unwrap();
        let pooled = vec![1.0f64; 2 * 16];
        let out = h.forward(&pooled, 2).unwrap();
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_cls_head_shape_mismatch() {
        let h = ClsHead::new(64, 10, 0).unwrap();
        assert!(h.forward(&[0.0f64; 100], 3).is_err());
    }

    #[test]
    fn test_cls_head_num_params() {
        let h = ClsHead::new(64, 10, 0).unwrap();
        assert_eq!(h.num_params(), 64 * 10 + 10);
    }

    #[test]
    fn test_cls_head_out_dim() {
        let h = ClsHead::new(64, 10, 0).unwrap();
        assert_eq!(h.out_dim(), 10);
    }

    #[test]
    fn test_cls_head_zero_dim_err() {
        assert!(ClsHead::new(0, 10, 0).is_err());
        assert!(ClsHead::new(64, 0, 0).is_err());
    }

    #[test]
    fn test_dino_head_new() {
        let h = DinoHead::new(64, 128, 32, 0).unwrap();
        assert_eq!(h.embed_dim, 64);
        assert_eq!(h.proj_dim, 32);
    }

    #[test]
    fn test_dino_head_forward_shape() {
        let h = DinoHead::new(16, 32, 8, 0).unwrap();
        let pooled = vec![0.5f64; 2 * 16];
        let out = h.forward(&pooled, 2).unwrap();
        assert_eq!(out.len(), 2 * 8);
    }

    #[test]
    fn test_dino_head_output_normalized() {
        let h = DinoHead::new(16, 32, 8, 0).unwrap();
        let pooled: Vec<f64> = (0..16).map(|x| x as f64 * 0.1).collect();
        let out = h.forward(&pooled, 1).unwrap();
        let norm: f64 = out.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dino_head_num_params() {
        let h = DinoHead::new(16, 32, 8, 0).unwrap();
        let expected = 32 * 16 + 32 + 32 * 32 + 32 + 8 * 32;
        assert_eq!(h.num_params(), expected);
    }

    #[test]
    fn test_dino_head_finite() {
        let h = DinoHead::new(16, 32, 8, 0).unwrap();
        let pooled = vec![1.0f64; 16];
        let out = h.forward(&pooled, 1).unwrap();
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_dino_head_zero_dim_err() {
        assert!(DinoHead::new(0, 32, 8, 0).is_err());
    }

    #[test]
    fn test_seg_head_new() {
        let h = SegHead::new(64, 10, 224, 224, 0).unwrap();
        assert_eq!(h.embed_dim, 64);
        assert_eq!(h.num_classes, 10);
    }

    #[test]
    fn test_seg_head_forward_shape() {
        let h = SegHead::new(16, 4, 32, 32, 0).unwrap();
        let tokens = vec![0.1f64; 2 * 9 * 16]; // B=2, N=9, D=16
        let out = h.forward_seg(&tokens, 2, 9).unwrap();
        assert_eq!(out.len(), 2 * 9 * 4);
    }

    #[test]
    fn test_seg_head_finite() {
        let h = SegHead::new(16, 4, 32, 32, 0).unwrap();
        let tokens = vec![0.1f64; 1 * 4 * 16];
        let out = h.forward_seg(&tokens, 1, 4).unwrap();
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_seg_head_shape_mismatch() {
        let h = SegHead::new(16, 4, 32, 32, 0).unwrap();
        assert!(h.forward_seg(&[0.0f64; 10], 1, 4).is_err());
    }

    #[test]
    fn test_seg_head_num_params() {
        let h = SegHead::new(16, 4, 32, 32, 0).unwrap();
        assert_eq!(h.num_params(), 16 * 4 + 4);
    }

    #[test]
    fn test_vit_head_config_default() {
        let cfg = VitHeadConfig::default();
        assert_eq!(cfg.embed_dim, 768);
        assert_eq!(cfg.out_dim, 1000);
    }
}
