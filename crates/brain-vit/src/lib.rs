//! # brain-vit — Production-Grade Vision Transformer
//!
//! A maximum-strength ViT implementation within the Brain deep learning
//! framework, containing all building blocks for:
//!
//! - **Patch embedding** ([`patch`])
//! - **Positional embeddings** ([`patch::pos_embed`])
//! - **Transformer blocks** ([`vit::blocks`])
//! - **Classification, DINO, and segmentation heads** ([`vit::heads`])
//! - **Full ViT model** ([`vit`])
//! - **Backbone presets** ([`backbones`])
//! - **Training utilities** ([`training`])
//! - **Object detection** ([`detection`])
//! - **Semantic segmentation** ([`segmentation`])
//! - **Model export** ([`export`])
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use brain_vit::r#impl::VitModel;
//! use brain_vit::config::VitConfig;
//!
//! let cfg = VitConfig::default();
//! let mut model = VitModel::new(cfg, 42).unwrap();
//! let pixels = vec![0.0f64; 3 * 224 * 224]; // 1 image
//! let output = model.forward(&pixels, 1).unwrap();
//! println!("Logits: {:?}", &output.logits[0][..5]);
//! ```
//!
//! ## Architecture Overview
//!
//! ```text
//!  [B, C, H, W]
//!      │
//!  PatchEmbed   ← projects to embed_dim
//!      │
//!  CLS token prepend + Position Embedding
//!      │
//!  [B, N+1, D]
//!      │
//!  TransformerBlock × depth
//!      │
//!  CLS pool / Mean pool
//!      │
//!  [B, D]  →  Head (linear)  →  [B, num_classes]
//! ```

// ── core infrastructure ──────────────────────────────────────────────────────
pub mod config;
pub mod core;
pub mod ops;
pub mod utils;

// ── impl: full forward-pass VitModel ─────────────────────────────────────────
pub mod r#impl;

// ── patch embedding modules ───────────────────────────────────────────────────
pub mod patch;

// ── vision transformer modules ────────────────────────────────────────────────
pub mod vit;

// ── backbone presets ──────────────────────────────────────────────────────────
pub mod backbones;

// ── downstream tasks ─────────────────────────────────────────────────────────
pub mod detection;
pub mod export;
pub mod segmentation;
pub mod training;

// ── crate-level re-exports ────────────────────────────────────────────────────
pub use backbones::{BackboneSize, VitBackbone};
pub use config::VitConfig;
pub use core::{SimpleRng, Tensor2D, Tensor3D, VitError, VitOutput, VitResult, VitState};
pub use detection::{iou, nms, Bbox, DetectionHead};
pub use export::{JsonCheckpoint, ModelCard, OnnxStub};
pub use patch::pos_embed::PosEmbed;
pub use patch::PatchEmbed;
pub use r#impl::VitModel;
pub use segmentation::{per_class_iou, pixel_accuracy, SegDecoder};
pub use training::{EarlyStopping, GradientClipper, LrScheduler, Optimizer, OptimizerConfig};
pub use vit::ViT;

/// Library version string.
pub const BRAIN_VIT_VERSION: &str = "1.0.0";

/// Default image size used across tests and demos.
pub const DEFAULT_IMAGE_SIZE: usize = 224;

/// Default patch size.
pub const DEFAULT_PATCH_SIZE: usize = 16;

/// Default embedding dimension for ViT-Base.
pub const DEFAULT_EMBED_DIM: usize = 768;

/// Default number of transformer blocks for ViT-Base.
pub const DEFAULT_DEPTH: usize = 12;

/// Default number of attention heads for ViT-Base.
pub const DEFAULT_NUM_HEADS: usize = 12;

/// Default number of ImageNet classes.
pub const DEFAULT_NUM_CLASSES: usize = 1000;

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn micro_vit() -> VitModel {
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
        VitModel::new(cfg, 42).unwrap()
    }

    #[test]
    fn test_version() {
        assert!(!BRAIN_VIT_VERSION.is_empty());
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_IMAGE_SIZE, 224);
        assert_eq!(DEFAULT_PATCH_SIZE, 16);
        assert_eq!(DEFAULT_EMBED_DIM, 768);
    }

    #[test]
    fn test_vitmodel_forward() {
        let mut m = micro_vit();
        let pixels = vec![0.5f64; 1 * 1 * 8 * 8];
        let out = m.forward(&pixels, 1).unwrap();
        assert_eq!(out.logits.len(), 1);
        assert_eq!(out.logits[0].len(), 4);
    }

    #[test]
    fn test_vitmodel_predict() {
        let mut m = micro_vit();
        let pixels = vec![0.5f64; 1 * 1 * 8 * 8];
        let probs = m.predict(&pixels, 1).unwrap();
        let sum: f64 = probs[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut m = micro_vit();
        let ckpt = m.save();
        let original_cls = m.cls_token.clone();
        m.cls_token.iter_mut().for_each(|x| *x = 0.0);
        m.load(&ckpt).unwrap();
        for (a, b) in original_cls.iter().zip(m.cls_token.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_patch_embed_shape() {
        let cfg = crate::config::PatchEmbedConfig {
            image_size: 16,
            patch_size: 4,
            in_channels: 1,
            embed_dim: 8,
            bias: true,
            mode: crate::config::PatchMode::Conv,
        };
        let pe = PatchEmbed::new(&cfg, 0).unwrap();
        let img = vec![0.5f64; 1 * 1 * 16 * 16];
        let tokens = pe.forward(&img, 1).unwrap();
        assert_eq!(tokens.len(), 16 * 8);
    }

    #[test]
    fn test_pos_embed_add() {
        let cfg = crate::config::PosEmbedConfig {
            seq_len: 5,
            embed_dim: 8,
            embed_type: crate::config::PosEmbedType::Sinusoidal,
            has_cls_token: true,
            grid_h: 2,
            grid_w: 2,
            dropout: 0.0,
        };
        let pe = PosEmbed::new(&cfg, 0).unwrap();
        let mut tokens = vec![0.0f64; 5 * 8];
        pe.add_to(&mut tokens, 1).unwrap();
        assert!(tokens.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_backbone_extract() {
        let bb = crate::backbones::micro_backbone(42).unwrap();
        let pixels = vec![0.5f64; 1 * 1 * 8 * 8];
        let cls = bb.extract_cls(&pixels, 1).unwrap();
        assert_eq!(cls.len(), 16);
        assert!(cls.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_optimizer_step() {
        let cfg = OptimizerConfig::default();
        let mut opt = Optimizer::new(cfg);
        let mut params = vec![1.0f64; 8];
        let grads = vec![0.01f64; 8];
        opt.step_params("w", &mut params, &grads).unwrap();
        assert!(params.iter().any(|&p| (p - 1.0).abs() > 1e-12));
    }

    #[test]
    fn test_lr_scheduler() {
        let mut sched = LrScheduler::new(
            1e-3,
            1e-5,
            100,
            10,
            crate::training::ScheduleType::CosineWithWarmup,
        );
        let lr = sched.step();
        assert!(lr >= 0.0 && lr <= 1e-3 + 1e-10);
    }

    #[test]
    fn test_early_stopping() {
        let mut es = EarlyStopping::new(2, 0.0, true);
        assert!(!es.update(1.0));
        assert!(!es.update(1.0));
        assert!(es.update(1.0));
    }

    #[test]
    fn test_gradient_clipper() {
        let clipper = GradientClipper::new(1.0).unwrap();
        let mut grads = vec![vec![3.0f64, 4.0f64]];
        let norm = clipper.clip(&mut grads);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_iou() {
        let b = Bbox::new(0.5, 0.5, 0.4, 0.4, 1.0, 0);
        assert!((iou(&b, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_nms() {
        let boxes = vec![
            Bbox::new(0.5, 0.5, 0.4, 0.4, 0.9, 0),
            Bbox::new(0.5, 0.5, 0.4, 0.4, 0.8, 0),
        ];
        let kept = nms(&boxes, 0.5);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn test_detection_head() {
        let h = DetectionHead::new(16, 4, 5, 0).unwrap();
        let q = vec![0.1f64; 1 * 5 * 16];
        let (boxes, cls) = h.forward(&q, 1).unwrap();
        assert_eq!(boxes.len(), 5 * 4);
        assert_eq!(cls.len(), 5 * 5);
    }

    #[test]
    fn test_seg_decoder() {
        let d = SegDecoder::new(16, 4, 0).unwrap();
        let tokens = vec![0.5f64; 1 * 9 * 16];
        let logits = d.forward(&tokens, 1, 9).unwrap();
        assert_eq!(logits.len(), 9 * 4);
    }

    #[test]
    fn test_pixel_accuracy() {
        let preds = vec![0usize, 1, 2];
        let gts = vec![0usize, 1, 2];
        let acc = pixel_accuracy(&preds, &gts).unwrap();
        assert!((acc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_per_class_iou() {
        let preds = vec![0usize, 1];
        let gts = vec![0usize, 1];
        let (_, miou) = per_class_iou(&preds, &gts, 2).unwrap();
        assert!((miou - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_json_checkpoint() {
        let mut w = std::collections::HashMap::new();
        w.insert("w".to_string(), vec![1.0f64, 2.0]);
        let ckpt = JsonCheckpoint::new("m", w);
        let text = ckpt.serialize();
        let loaded = JsonCheckpoint::deserialize(&text).unwrap();
        assert_eq!(loaded.total_params(), 2);
    }

    #[test]
    fn test_onnx_stub() {
        let stub = OnnxStub::from_vit("ViT-T", 8, 1, 4, 16, 1, 2);
        assert!(!stub.describe().is_empty());
    }

    #[test]
    fn test_model_card() {
        let mut card = ModelCard::for_vit("ViT-B", 224, 16, 768, 12, 12, 1000);
        card.add_metric("top1", 0.812);
        assert!(card.to_markdown().contains("top1"));
    }

    #[test]
    fn test_vit_error_display() {
        let e = VitError::EmptyBatch;
        assert!(!format!("{}", e).is_empty());
    }

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(0);
        let v = rng.gen_vec(10, 0.0, 1.0);
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_tensor2d_matmul() {
        let a = Tensor2D::from_data(2, 3, vec![1.0; 6]).unwrap();
        let b = Tensor2D::from_data(3, 2, vec![1.0; 6]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
    }

    #[test]
    fn test_tensor3d_pool() {
        let t = Tensor3D::from_data(2, 5, 8, vec![1.0f64; 80]).unwrap();
        let pooled = t.mean_pool();
        assert_eq!(pooled.rows, 2);
        assert_eq!(pooled.cols, 8);
    }

    #[test]
    fn test_brain_vit_version_semver() {
        let parts: Vec<&str> = BRAIN_VIT_VERSION.split('.').collect();
        assert_eq!(parts.len(), 3);
    }
}
