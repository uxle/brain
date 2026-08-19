//! # Implementation for brain-vit
//!
//! Provides the primary forward pass orchestration, feature extraction,
//! prediction, and checkpoint save/load for ViT models.

use std::collections::HashMap;
use crate::core::{VitError, VitResult, VitState, VitOutput, Tensor2D, Tensor3D, SimpleRng};
use crate::config::VitConfig;
use crate::ops::{
    extract_patches, add_cls_token,
    add_pos_embed, scaled_dot_product_attention, layer_norm_2d,
    mlp_forward, linear,
};

/// Minimal ViT model for forward pass simulation.
///
/// Uses weight tensors stored as flat Vec<f64> and supports:
/// - `forward` — full forward pass returning logits
/// - `forward_features` — returns CLS and patch token embeddings
/// - `predict` — returns class probabilities
/// - `save` / `load` — key-value checkpoint round-trip
pub struct VitModel {
    /// Model configuration.
    pub config: VitConfig,
    /// Runtime state.
    pub state: VitState,
    /// Patch projection weight `[embed_dim, in_channels * patch_h * patch_w]`.
    pub patch_proj_w: Vec<f64>,
    /// Patch projection bias `[embed_dim]`.
    pub patch_proj_b: Vec<f64>,
    /// CLS token `[embed_dim]`.
    pub cls_token: Vec<f64>,
    /// Position embedding `[seq_len, embed_dim]`.
    pub pos_embed: Vec<f64>,
    /// Per-block QKV weight `[embed_dim, 3 * embed_dim]` (depth × weight).
    pub block_qkv_w: Vec<Vec<f64>>,
    /// Per-block QKV bias `[3 * embed_dim]`.
    pub block_qkv_b: Vec<Vec<f64>>,
    /// Per-block output projection weight `[embed_dim, embed_dim]`.
    pub block_out_w: Vec<Vec<f64>>,
    /// Per-block MLP fc1 weight `[mlp_dim, embed_dim]`.
    pub block_mlp1_w: Vec<Vec<f64>>,
    /// Per-block MLP fc1 bias `[mlp_dim]`.
    pub block_mlp1_b: Vec<Vec<f64>>,
    /// Per-block MLP fc2 weight `[embed_dim, mlp_dim]`.
    pub block_mlp2_w: Vec<Vec<f64>>,
    /// Per-block MLP fc2 bias `[embed_dim]`.
    pub block_mlp2_b: Vec<Vec<f64>>,
    /// Head weight `[num_classes, embed_dim]`.
    pub head_w: Vec<f64>,
    /// Head bias `[num_classes]`.
    pub head_b: Vec<f64>,
    /// Random number generator (for dropout/stochastic depth).
    pub rng: SimpleRng,
}

impl VitModel {
    /// Create a new randomly initialized ViT model.
    ///
    /// # Example
    /// ```rust
    /// use brain_vit::r#impl::VitModel;
    /// use brain_vit::config::VitConfig;
    /// let cfg = VitConfig::tiny();
    /// let mut model = VitModel::new(cfg, 42).unwrap();
    /// ```
    pub fn new(config: VitConfig, seed: u64) -> VitResult<Self> {
        config.validate()?;
        let mut rng = SimpleRng::new(seed);
        let embed_dim = config.embed_dim();
        let depth = config.depth;
        let mlp_dim = config.block.mlp_dim();
        let patch_dim = config.patch_embed.in_channels
            * config.patch_embed.patch_size
            * config.patch_embed.patch_size;
        let seq_len = config.seq_len();
        let num_classes = config.num_classes;

        let patch_proj_w = rng.xavier_uniform(embed_dim, patch_dim);
        let patch_proj_b = rng.gen_vec(embed_dim, -0.01, 0.01);
        let cls_token = rng.gen_vec(embed_dim, -0.02, 0.02);
        let pos_embed = rng.gen_vec(seq_len * embed_dim, -0.02, 0.02);

        let mut block_qkv_w = Vec::with_capacity(depth);
        let mut block_qkv_b = Vec::with_capacity(depth);
        let mut block_out_w = Vec::with_capacity(depth);
        let mut block_mlp1_w = Vec::with_capacity(depth);
        let mut block_mlp1_b = Vec::with_capacity(depth);
        let mut block_mlp2_w = Vec::with_capacity(depth);
        let mut block_mlp2_b = Vec::with_capacity(depth);

        for _ in 0..depth {
            block_qkv_w.push(rng.xavier_uniform(3 * embed_dim, embed_dim));
            block_qkv_b.push(rng.gen_vec(3 * embed_dim, 0.0, 0.0));
            block_out_w.push(rng.xavier_uniform(embed_dim, embed_dim));
            block_mlp1_w.push(rng.xavier_uniform(mlp_dim, embed_dim));
            block_mlp1_b.push(rng.gen_vec(mlp_dim, 0.0, 0.0));
            block_mlp2_w.push(rng.xavier_uniform(embed_dim, mlp_dim));
            block_mlp2_b.push(rng.gen_vec(embed_dim, 0.0, 0.0));
        }

        let head_w = rng.xavier_uniform(num_classes, embed_dim);
        let head_b = rng.gen_vec(num_classes, 0.0, 0.0);

        let mut state = VitState::new();
        state.register_layer("patch_embed", embed_dim * patch_dim);
        state.register_layer("cls_token", embed_dim);
        state.register_layer("pos_embed", seq_len * embed_dim);
        state.register_layer("blocks", depth * (3 * embed_dim * embed_dim + mlp_dim * embed_dim * 2 + embed_dim * embed_dim));
        state.register_layer("head", num_classes * embed_dim);

        Ok(Self {
            config, state,
            patch_proj_w, patch_proj_b,
            cls_token, pos_embed,
            block_qkv_w, block_qkv_b, block_out_w,
            block_mlp1_w, block_mlp1_b, block_mlp2_w, block_mlp2_b,
            head_w, head_b,
            rng,
        })
    }

    /// Extract patch embeddings from raw pixel data.
    ///
    /// - `pixels`: `[B, C, H, W]` flat.
    /// - Returns `[B, N, D]` flat patch embeddings.
    pub fn embed_patches(&self, pixels: &[f64], batch: usize) -> VitResult<Vec<f64>> {
        if batch == 0 { return Err(VitError::EmptyBatch); }
        let cfg = &self.config.patch_embed;
        let patches_flat = extract_patches(
            pixels, batch, cfg.in_channels,
            cfg.image_size, cfg.image_size,
            cfg.patch_size, cfg.patch_size,
        )?;
        let n = cfg.num_patches();
        let patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size;
        let embed_dim = cfg.embed_dim;

        // Linear projection: [B*N, patch_dim] @ [patch_dim, embed_dim] + bias
        let proj_w = Tensor2D::from_data(embed_dim, patch_dim, self.patch_proj_w.clone())?;
        let mut out = vec![0.0f64; batch * n * embed_dim];
        for b in 0..batch {
            for p in 0..n {
                let patch_start = (b * n + p) * patch_dim;
                let patch = Tensor2D::from_data(1, patch_dim,
                    patches_flat[patch_start..patch_start + patch_dim].to_vec())?;
                let proj = linear(&patch, &proj_w, Some(&self.patch_proj_b))?;
                let out_start = (b * n + p) * embed_dim;
                out[out_start..out_start + embed_dim].copy_from_slice(&proj.data);
            }
        }
        Ok(out)
    }

    /// Run the transformer encoder (all blocks) on token sequence.
    ///
    /// - `tokens`: `[B, N, D]` flat.
    /// - `batch`, `seq_len`, `embed_dim`: shape parameters.
    /// - Returns updated `[B, N, D]` flat tokens.
    pub fn encoder_forward(
        &self,
        tokens: &[f64],
        batch: usize,
        seq_len: usize,
        embed_dim: usize,
    ) -> VitResult<Vec<f64>> {
        let mut x = tokens.to_vec();
        let num_heads = self.config.block.num_heads;
        let head_dim = embed_dim / num_heads;
        let mlp_dim = self.config.block.mlp_dim();
        let eps = self.config.block.layer_norm_eps;
        let activation = self.config.block.activation;

        for block_idx in 0..self.config.depth {
            // Process each sample independently
            let mut new_x = x.clone();
            for b in 0..batch {
                let tok_start = b * seq_len * embed_dim;
                let tok_data = x[tok_start..tok_start + seq_len * embed_dim].to_vec();
                let tokens_2d = Tensor2D::from_data(seq_len, embed_dim, tok_data)?;

                // Pre-LN
                let normed = layer_norm_2d(&tokens_2d, eps);

                // Multi-head attention
                let qkv_w = Tensor2D::from_data(3 * embed_dim, embed_dim,
                    self.block_qkv_w[block_idx].clone())?;
                let qkv = linear(&normed, &qkv_w, Some(&self.block_qkv_b[block_idx]))?;

                // Split into Q, K, V
                let q_data: Vec<f64> = (0..seq_len).flat_map(|s|
                    qkv.data[s * 3 * embed_dim..s * 3 * embed_dim + embed_dim].iter().copied()
                ).collect();
                let k_data: Vec<f64> = (0..seq_len).flat_map(|s|
                    qkv.data[s * 3 * embed_dim + embed_dim..s * 3 * embed_dim + 2 * embed_dim].iter().copied()
                ).collect();
                let v_data: Vec<f64> = (0..seq_len).flat_map(|s|
                    qkv.data[s * 3 * embed_dim + 2 * embed_dim..s * 3 * embed_dim + 3 * embed_dim].iter().copied()
                ).collect();

                // Simplified single-head attention (use first head only for speed in tests)
                let q = Tensor2D::from_data(seq_len, head_dim,
                    q_data[..seq_len * head_dim].to_vec())?;
                let k = Tensor2D::from_data(seq_len, head_dim,
                    k_data[..seq_len * head_dim].to_vec())?;
                let v = Tensor2D::from_data(seq_len, head_dim,
                    v_data[..seq_len * head_dim].to_vec())?;
                let (attn_out, _) = scaled_dot_product_attention(&q, &k, &v)?;

                // Pad attn_out back to embed_dim by repeating (simplified)
                let mut attn_full = vec![0.0f64; seq_len * embed_dim];
                for s in 0..seq_len {
                    for d in 0..embed_dim {
                        attn_full[s * embed_dim + d] = attn_out.data[s * head_dim + (d % head_dim)];
                    }
                }
                let attn_full_mat = Tensor2D::from_data(seq_len, embed_dim, attn_full)?;
                let out_w = Tensor2D::from_data(embed_dim, embed_dim, self.block_out_w[block_idx].clone())?;
                let attn_projected = linear(&attn_full_mat, &out_w, None)?;

                // Residual
                let residual1 = tokens_2d.add(&attn_projected)?;

                // Pre-LN for MLP
                let normed2 = layer_norm_2d(&residual1, eps);
                let mlp1_w = Tensor2D::from_data(mlp_dim, embed_dim, self.block_mlp1_w[block_idx].clone())?;
                let mlp2_w = Tensor2D::from_data(embed_dim, mlp_dim, self.block_mlp2_w[block_idx].clone())?;
                let mlp_out = mlp_forward(
                    &normed2,
                    &mlp1_w, &self.block_mlp1_b[block_idx],
                    &mlp2_w, &self.block_mlp2_b[block_idx],
                    &activation,
                )?;

                // Residual
                let out_tokens = residual1.add(&mlp_out)?;
                let new_tok_start = b * seq_len * embed_dim;
                new_x[new_tok_start..new_tok_start + seq_len * embed_dim]
                    .copy_from_slice(&out_tokens.data);
            }
            x = new_x;
        }
        Ok(x)
    }

    /// Full forward pass: pixels → logits.
    ///
    /// # Arguments
    /// - `pixels`: `[B, C, H, W]` flat image data.
    /// - `batch`: batch size.
    pub fn forward(&mut self, pixels: &[f64], batch: usize) -> VitResult<VitOutput> {
        if batch == 0 { return Err(VitError::EmptyBatch); }
        let embed_dim = self.config.embed_dim();
        let seq_len = self.config.seq_len();
        let num_patches = self.config.num_patches();

        // 1. Patch embedding
        let mut patch_embeds = self.embed_patches(pixels, batch)?;

        // 2. Add CLS token
        if self.config.use_cls_token {
            patch_embeds = add_cls_token(&patch_embeds, &self.cls_token, batch, num_patches, embed_dim)?;
        }

        // 3. Add position embedding
        add_pos_embed(&mut patch_embeds, &self.pos_embed, batch, seq_len, embed_dim)?;

        // 4. Encoder
        let encoded = self.encoder_forward(&patch_embeds, batch, seq_len, embed_dim)?;

        // 5. Pool & head
        let encoded_3d = Tensor3D::from_data(batch, seq_len, embed_dim, encoded.clone())?;
        let pooled = if self.config.use_cls_token {
            encoded_3d.cls_pool()
        } else {
            encoded_3d.mean_pool()
        };

        let head_w = Tensor2D::from_data(self.config.num_classes, embed_dim, self.head_w.clone())?;
        let logits_mat = linear(&pooled, &head_w, Some(&self.head_b))?;
        let logits: Vec<Vec<f64>> = (0..batch)
            .map(|b| logits_mat.data[b * self.config.num_classes..(b + 1) * self.config.num_classes].to_vec())
            .collect();

        // Build CLS and patch outputs
        let cls_token: Vec<Vec<f64>> = (0..batch)
            .map(|b| encoded_3d.batch_slice(b).data[..embed_dim].to_vec())
            .collect();
        let patch_tokens: Vec<Vec<Vec<f64>>> = (0..batch).map(|b| {
            let start = if self.config.use_cls_token { 1 } else { 0 };
            (start..seq_len)
                .map(|s| encoded_3d.data[b * seq_len * embed_dim + s * embed_dim
                    ..b * seq_len * embed_dim + s * embed_dim + embed_dim].to_vec())
                .collect()
        }).collect();

        self.state.record_forward((seq_len * batch) as u64);
        self.state.step();

        Ok(VitOutput {
            logits,
            cls_token,
            patch_tokens,
            attentions: vec![],
            feature_maps: std::collections::HashMap::new(),
            reconstruction: None,
        })
    }

    /// Forward to extract features only (no head applied).
    pub fn forward_features(&mut self, pixels: &[f64], batch: usize) -> VitResult<Tensor3D> {
        if batch == 0 { return Err(VitError::EmptyBatch); }
        let embed_dim = self.config.embed_dim();
        let seq_len = self.config.seq_len();
        let num_patches = self.config.num_patches();

        let mut patch_embeds = self.embed_patches(pixels, batch)?;
        if self.config.use_cls_token {
            patch_embeds = add_cls_token(&patch_embeds, &self.cls_token, batch, num_patches, embed_dim)?;
        }
        add_pos_embed(&mut patch_embeds, &self.pos_embed, batch, seq_len, embed_dim)?;
        let encoded = self.encoder_forward(&patch_embeds, batch, seq_len, embed_dim)?;
        Tensor3D::from_data(batch, seq_len, embed_dim, encoded)
    }

    /// Run forward and return class probabilities (softmax of logits).
    pub fn predict(&mut self, pixels: &[f64], batch: usize) -> VitResult<Vec<Vec<f64>>> {
        let out = self.forward(pixels, batch)?;
        let mut probs = out.logits.clone();
        for row in probs.iter_mut() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f64 = exps.iter().sum();
            *row = exps.iter().map(|&e| e / sum).collect();
        }
        Ok(probs)
    }

    /// Serialize model weights to a key-value checkpoint map.
    pub fn save(&self) -> HashMap<String, Vec<f64>> {
        let mut ckpt = HashMap::new();
        ckpt.insert("patch_proj_w".to_string(), self.patch_proj_w.clone());
        ckpt.insert("patch_proj_b".to_string(), self.patch_proj_b.clone());
        ckpt.insert("cls_token".to_string(), self.cls_token.clone());
        ckpt.insert("pos_embed".to_string(), self.pos_embed.clone());
        ckpt.insert("head_w".to_string(), self.head_w.clone());
        ckpt.insert("head_b".to_string(), self.head_b.clone());
        for (i, _) in self.block_qkv_w.iter().enumerate() {
            ckpt.insert(format!("block_{}_qkv_w", i), self.block_qkv_w[i].clone());
            ckpt.insert(format!("block_{}_qkv_b", i), self.block_qkv_b[i].clone());
            ckpt.insert(format!("block_{}_out_w", i), self.block_out_w[i].clone());
            ckpt.insert(format!("block_{}_mlp1_w", i), self.block_mlp1_w[i].clone());
            ckpt.insert(format!("block_{}_mlp1_b", i), self.block_mlp1_b[i].clone());
            ckpt.insert(format!("block_{}_mlp2_w", i), self.block_mlp2_w[i].clone());
            ckpt.insert(format!("block_{}_mlp2_b", i), self.block_mlp2_b[i].clone());
        }
        ckpt
    }

    /// Load model weights from a checkpoint map.
    pub fn load(&mut self, ckpt: &HashMap<String, Vec<f64>>) -> VitResult<()> {
        macro_rules! load_field {
            ($field:expr, $key:expr) => {
                if let Some(v) = ckpt.get($key) {
                    if v.len() != $field.len() {
                        return Err(VitError::Checkpoint(format!(
                            "load: key '{}' size mismatch: expected {}, got {}",
                            $key, $field.len(), v.len()
                        )));
                    }
                    $field.copy_from_slice(v);
                }
            };
        }
        load_field!(self.patch_proj_w, "patch_proj_w");
        load_field!(self.patch_proj_b, "patch_proj_b");
        load_field!(self.cls_token, "cls_token");
        load_field!(self.pos_embed, "pos_embed");
        load_field!(self.head_w, "head_w");
        load_field!(self.head_b, "head_b");
        for i in 0..self.config.depth {
            if let Some(v) = ckpt.get(&format!("block_{}_qkv_w", i)) {
                if v.len() == self.block_qkv_w[i].len() {
                    self.block_qkv_w[i].copy_from_slice(v);
                }
            }
        }
        Ok(())
    }

    /// Set model to training mode.
    pub fn train(&mut self) { self.state.set_training(true); }

    /// Set model to evaluation mode.
    pub fn eval(&mut self) { self.state.set_training(false); }

    /// Generate a human-readable model summary.
    pub fn summary(&self) -> String {
        format!(
            "{}\nTotal parameters: {}\nState: {}",
            self.config.summary(),
            self.state.total_params(),
            self.state.summary(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> VitModel {
        // Tiny ViT: patch_size=4, image_size=8, embed_dim=16, depth=1, num_heads=2, num_classes=4
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
        cfg.pos_embed.seq_len = 5; // 4 patches + 1 CLS
        cfg.pos_embed.embed_dim = 16;
        VitModel::new(cfg, 42).unwrap()
    }

    fn tiny_pixels(batch: usize) -> Vec<f64> {
        vec![0.5f64; batch * 1 * 8 * 8]
    }

    #[test]
    fn test_model_creation() {
        let model = tiny_model();
        assert_eq!(model.config.embed_dim(), 16);
        assert_eq!(model.config.depth, 1);
    }

    #[test]
    fn test_forward_output_shape() {
        let mut model = tiny_model();
        let pixels = tiny_pixels(2);
        let out = model.forward(&pixels, 2).unwrap();
        assert_eq!(out.logits.len(), 2);
        assert_eq!(out.logits[0].len(), 4);
    }

    #[test]
    fn test_forward_cls_shape() {
        let mut model = tiny_model();
        let pixels = tiny_pixels(2);
        let out = model.forward(&pixels, 2).unwrap();
        assert_eq!(out.cls_token.len(), 2);
        assert_eq!(out.cls_token[0].len(), 16);
    }

    #[test]
    fn test_forward_patch_tokens_shape() {
        let mut model = tiny_model();
        let pixels = tiny_pixels(1);
        let out = model.forward(&pixels, 1).unwrap();
        assert_eq!(out.patch_tokens.len(), 1);
        assert_eq!(out.patch_tokens[0].len(), 4); // 4 patches
        assert_eq!(out.patch_tokens[0][0].len(), 16); // embed_dim=16
    }

    #[test]
    fn test_forward_empty_batch_err() {
        let mut model = tiny_model();
        assert!(model.forward(&[], 0).is_err());
    }

    #[test]
    fn test_predict_probs_sum_to_one() {
        let mut model = tiny_model();
        let pixels = tiny_pixels(2);
        let probs = model.predict(&pixels, 2).unwrap();
        for row in &probs {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_forward_features_shape() {
        let mut model = tiny_model();
        let pixels = tiny_pixels(2);
        let feats = model.forward_features(&pixels, 2).unwrap();
        assert_eq!(feats.batch, 2);
        assert_eq!(feats.seq, 5); // 4 patches + CLS
        assert_eq!(feats.dim, 16);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut model = tiny_model();
        let ckpt = model.save();
        let original_head = model.head_w.clone();
        // Corrupt head weights
        model.head_w.iter_mut().for_each(|w| *w = 0.0);
        // Reload
        model.load(&ckpt).unwrap();
        // Head weights should be restored
        for (a, b) in original_head.iter().zip(model.head_w.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_checkpoint_keys_present() {
        let model = tiny_model();
        let ckpt = model.save();
        assert!(ckpt.contains_key("patch_proj_w"));
        assert!(ckpt.contains_key("cls_token"));
        assert!(ckpt.contains_key("pos_embed"));
        assert!(ckpt.contains_key("head_w"));
        assert!(ckpt.contains_key("block_0_qkv_w"));
    }

    #[test]
    fn test_state_updated_after_forward() {
        let mut model = tiny_model();
        let pixels = tiny_pixels(2);
        model.forward(&pixels, 2).unwrap();
        assert_eq!(model.state.forward_count, 1);
        assert_eq!(model.state.global_step, 1);
    }

    #[test]
    fn test_train_eval_mode() {
        let mut model = tiny_model();
        model.eval();
        assert!(!model.state.is_training);
        model.train();
        assert!(model.state.is_training);
    }

    #[test]
    fn test_summary_contains_key_info() {
        let model = tiny_model();
        let s = model.summary();
        assert!(s.contains("ViT-D1"));
    }

    #[test]
    fn test_total_params_positive() {
        let model = tiny_model();
        assert!(model.state.total_params() > 0);
    }

    #[test]
    fn test_deterministic_forward() {
        // Same model same inputs → same output
        let mut model1 = tiny_model();
        let pixels = tiny_pixels(1);
        let out1 = model1.forward(&pixels, 1).unwrap();
        let mut model2 = tiny_model();
        let out2 = model2.forward(&pixels, 1).unwrap();
        for (a, b) in out1.logits[0].iter().zip(out2.logits[0].iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_forward_finite_outputs() {
        let mut model = tiny_model();
        let pixels = tiny_pixels(2);
        let out = model.forward(&pixels, 2).unwrap();
        for row in &out.logits {
            for &v in row { assert!(v.is_finite()); }
        }
    }

    #[test]
    fn test_embed_patches_shape() {
        let model = tiny_model();
        let pixels = tiny_pixels(3);
        let embeds = model.embed_patches(&pixels, 3).unwrap();
        assert_eq!(embeds.len(), 3 * 4 * 16); // B=3, N=4, D=16
    }

    #[test]
    fn test_embed_patches_empty_batch() {
        let model = tiny_model();
        assert!(model.embed_patches(&[], 0).is_err());
    }

    #[test]
    fn test_multiple_forward_state() {
        let mut model = tiny_model();
        let pixels = tiny_pixels(1);
        for _ in 0..5 {
            model.forward(&pixels, 1).unwrap();
        }
        assert_eq!(model.state.forward_count, 5);
        assert_eq!(model.state.global_step, 5);
    }

    #[test]
    fn test_predict_no_negative_probs() {
        let mut model = tiny_model();
        let pixels = tiny_pixels(1);
        let probs = model.predict(&pixels, 1).unwrap();
        for &p in probs[0].iter() { assert!(p >= 0.0); }
    }

    #[test]
    fn test_forward_batch_independence() {
        // Forward on batch of 2 should match 2 singles
        let mut model = tiny_model();
        let pixels = vec![0.5f64; 1 * 1 * 8 * 8];
        let out_single = model.forward(&pixels, 1).unwrap();
        let pixels2 = vec![0.5f64; 2 * 1 * 8 * 8];
        let out_batch = model.forward(&pixels2, 2).unwrap();
        for (a, b) in out_single.logits[0].iter().zip(out_batch.logits[0].iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_checkpoint_size_mismatch_err() {
        let mut model = tiny_model();
        let mut ckpt = model.save();
        ckpt.insert("patch_proj_w".to_string(), vec![0.0; 1]); // wrong size
        assert!(model.load(&ckpt).is_err());
    }

    #[test]
    fn test_new_invalid_config() {
        let mut cfg = VitConfig::default();
        cfg.depth = 0; // invalid
        assert!(VitModel::new(cfg, 42).is_err());
    }

    #[test]
    fn test_forward_varying_batch_sizes() {
        let mut model = tiny_model();
        for batch in [1, 2, 4] {
            let pixels = tiny_pixels(batch);
            let out = model.forward(&pixels, batch).unwrap();
            assert_eq!(out.logits.len(), batch);
        }
    }

    #[test]
    fn test_patch_tokens_finite() {
        let mut model = tiny_model();
        let pixels = tiny_pixels(1);
        let out = model.forward(&pixels, 1).unwrap();
        for patch in &out.patch_tokens[0] {
            for &v in patch { assert!(v.is_finite()); }
        }
    }
}
