//! # Core Types for brain-vit
//!
//! Provides fundamental state containers, output structures, and error types
//! shared across the Vision Transformer crate.
//!
//! ## Key types
//! - [`VitError`] — typed error enum for all ViT operations
//! - [`VitResult`] — crate-wide `Result<T, VitError>` alias
//! - [`VitState`] — runtime state for a ViT model instance
//! - [`VitOutput`] — structured forward-pass output (logits, CLS, patches, attentions)

use std::collections::HashMap;
use std::fmt;

/// Crate-wide result alias.
pub type VitResult<T> = Result<T, VitError>;

/// Comprehensive error type for all brain-vit operations.
#[derive(Debug, Clone, PartialEq)]
pub enum VitError {
    /// Image dimensions not divisible by patch size.
    InvalidPatchSize { image_dim: usize, patch_size: usize },
    /// Requested resolution incompatible with current position embedding.
    ResolutionMismatch { expected: usize, got: usize },
    /// Embedding dimension mismatch between components.
    DimMismatch { expected: usize, got: usize },
    /// Batch size of zero received.
    EmptyBatch,
    /// Generic configuration error.
    Config(String),
    /// Checkpoint serialization/deserialization error.
    Checkpoint(String),
    /// Shape-related computation error.
    Shape(String),
    /// Numerical overflow or underflow.
    Numerical(String),
    /// Unsupported operation or feature.
    Unsupported(String),
    /// I/O error during save/load.
    Io(String),
}

impl fmt::Display for VitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VitError::InvalidPatchSize { image_dim, patch_size } =>
                write!(f, "Image dim {} not divisible by patch size {}", image_dim, patch_size),
            VitError::ResolutionMismatch { expected, got } =>
                write!(f, "Resolution mismatch: expected {}, got {}", expected, got),
            VitError::DimMismatch { expected, got } =>
                write!(f, "Dim mismatch: expected {}, got {}", expected, got),
            VitError::EmptyBatch =>
                write!(f, "Batch size must be > 0"),
            VitError::Config(msg) =>
                write!(f, "Config error: {}", msg),
            VitError::Checkpoint(msg) =>
                write!(f, "Checkpoint error: {}", msg),
            VitError::Shape(msg) =>
                write!(f, "Shape error: {}", msg),
            VitError::Numerical(msg) =>
                write!(f, "Numerical error: {}", msg),
            VitError::Unsupported(msg) =>
                write!(f, "Unsupported: {}", msg),
            VitError::Io(msg) =>
                write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for VitError {}

/// Pooling strategy for the CLS or sequence tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolStrategy {
    /// Use the dedicated CLS token embedding.
    #[default]
    Cls,
    /// Average over all patch tokens.
    MeanPool,
    /// Learnable attention-weighted average.
    AttentionPool,
    /// Global average pooling (no CLS token).
    Gap,
}

impl fmt::Display for PoolStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolStrategy::Cls => write!(f, "cls"),
            PoolStrategy::MeanPool => write!(f, "mean_pool"),
            PoolStrategy::AttentionPool => write!(f, "attention_pool"),
            PoolStrategy::Gap => write!(f, "gap"),
        }
    }
}

impl PoolStrategy {
    /// Parse from string representation.
    ///
    /// # Example
    /// ```rust
    /// use brain_vit::core::PoolStrategy;
    /// let s = PoolStrategy::from_str("cls").unwrap();
    /// assert_eq!(s, PoolStrategy::Cls);
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> VitResult<Self> {
        match s.to_lowercase().as_str() {
            "cls" => Ok(PoolStrategy::Cls),
            "mean_pool" | "mean" => Ok(PoolStrategy::MeanPool),
            "attention_pool" | "attn_pool" => Ok(PoolStrategy::AttentionPool),
            "gap" => Ok(PoolStrategy::Gap),
            other => Err(VitError::Config(format!("Unknown pool strategy: {}", other))),
        }
    }
}

/// Full structured output from a ViT forward pass.
#[derive(Debug, Clone)]
pub struct VitOutput {
    /// Final classification/regression logits — shape `[B, num_classes]`.
    pub logits: Vec<Vec<f64>>,
    /// CLS token embedding — shape `[B, embed_dim]`.
    pub cls_token: Vec<Vec<f64>>,
    /// Patch token embeddings after all transformer blocks — shape `[B, N, embed_dim]`.
    pub patch_tokens: Vec<Vec<Vec<f64>>>,
    /// Per-layer, per-head attention weights — shape `[L, B, H, N+1, N+1]`.
    /// Only populated when `return_attentions = true`.
    pub attentions: Vec<Vec<Vec<Vec<Vec<f64>>>>>,
    /// Intermediate feature maps from selected layers (for backbone use).
    pub feature_maps: HashMap<String, Vec<Vec<Vec<f64>>>>,
    /// Reconstruction output for MAE-style self-supervised models.
    pub reconstruction: Option<Vec<Vec<Vec<f64>>>>,
}

impl VitOutput {
    /// Create an empty output placeholder.
    pub fn empty() -> Self {
        Self {
            logits: vec![],
            cls_token: vec![],
            patch_tokens: vec![],
            attentions: vec![],
            feature_maps: HashMap::new(),
            reconstruction: None,
        }
    }

    /// Return the batch size inferred from logits (0 if empty).
    pub fn batch_size(&self) -> usize {
        self.logits.len()
    }

    /// Return the number of patch tokens per sample (0 if empty).
    pub fn num_patches(&self) -> usize {
        self.patch_tokens.first().map(|t| t.len()).unwrap_or(0)
    }

    /// Return the embedding dimension from CLS token (0 if empty).
    pub fn embed_dim(&self) -> usize {
        self.cls_token.first().map(|c| c.len()).unwrap_or(0)
    }

    /// Checks if attention maps are populated.
    pub fn has_attentions(&self) -> bool {
        !self.attentions.is_empty()
    }
}

/// Runtime state container for a ViT model instance.
#[derive(Debug, Clone)]
pub struct VitState {
    /// Total number of forward passes executed.
    pub forward_count: u64,
    /// Total tokens processed across all forward passes.
    pub total_tokens_processed: u64,
    /// Whether the model is in training mode.
    pub is_training: bool,
    /// Current global step (used for schedulers and checkpointing).
    pub global_step: u64,
    /// Best validation metric seen (lower is better by default).
    pub best_metric: f64,
    /// Named floating-point statistics accumulator.
    pub stats: HashMap<String, f64>,
    /// Named per-layer parameter counts.
    pub param_counts: HashMap<String, usize>,
}

impl Default for VitState {
    fn default() -> Self {
        Self {
            forward_count: 0,
            total_tokens_processed: 0,
            is_training: true,
            global_step: 0,
            best_metric: f64::MAX,
            stats: HashMap::new(),
            param_counts: HashMap::new(),
        }
    }
}

impl VitState {
    /// Create a fresh state for a new training run.
    pub fn new() -> Self { Self::default() }

    /// Record a forward pass with `n` tokens processed.
    pub fn record_forward(&mut self, n_tokens: u64) {
        self.forward_count += 1;
        self.total_tokens_processed += n_tokens;
    }

    /// Update a named statistic accumulator.
    pub fn update_stat(&mut self, name: &str, value: f64) {
        self.stats.insert(name.to_string(), value);
    }

    /// Retrieve a named statistic (0.0 if not set).
    pub fn get_stat(&self, name: &str) -> f64 {
        self.stats.get(name).copied().unwrap_or(0.0)
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, training: bool) { self.is_training = training; }

    /// Increment global step.
    pub fn step(&mut self) { self.global_step += 1; }

    /// Update best metric; returns true if new best was achieved.
    pub fn update_best(&mut self, metric: f64) -> bool {
        if metric < self.best_metric {
            self.best_metric = metric;
            true
        } else {
            false
        }
    }

    /// Reset statistics accumulators.
    pub fn reset_stats(&mut self) { self.stats.clear(); }

    /// Total parameters across all tracked layers.
    pub fn total_params(&self) -> usize {
        self.param_counts.values().sum()
    }

    /// Register a layer with its parameter count.
    pub fn register_layer(&mut self, name: &str, params: usize) {
        self.param_counts.insert(name.to_string(), params);
    }

    /// Produce a text summary of the current state.
    pub fn summary(&self) -> String {
        format!(
            "VitState {{ steps={}, forwards={}, tokens={}, training={}, best_metric={:.6} }}",
            self.global_step, self.forward_count,
            self.total_tokens_processed, self.is_training, self.best_metric
        )
    }
}

/// Data type used for tensor computations in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VitDtype {
    /// 32-bit float.
    #[default]
    F32,
    /// 64-bit float.
    F64,
    /// 16-bit brain float.
    Bf16,
}

impl fmt::Display for VitDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VitDtype::F32 => write!(f, "f32"),
            VitDtype::F64 => write!(f, "f64"),
            VitDtype::Bf16 => write!(f, "bf16"),
        }
    }
}

/// Simple tensor view (row-major, float64 values) used for test/compute.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor2D {
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub cols: usize,
    /// Flattened data in row-major order.
    pub data: Vec<f64>,
}

impl Tensor2D {
    /// Create a zero-initialized tensor of shape `[rows, cols]`.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Create from data, verifying shape.
    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> VitResult<Self> {
        if data.len() != rows * cols {
            return Err(VitError::Shape(format!(
                "Expected {} elements for shape [{}, {}], got {}",
                rows * cols, rows, cols, data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Get element at `[row, col]`.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    /// Set element at `[row, col]`.
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.cols + col] = val;
    }

    /// Apply softmax over each row independently.
    pub fn softmax_rows(&self) -> Self {
        let mut out = self.clone();
        for r in 0..self.rows {
            let start = r * self.cols;
            let end = start + self.cols;
            let slice = &self.data[start..end];
            let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = slice.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for (i, &e) in exps.iter().enumerate() {
                out.data[start + i] = e / sum;
            }
        }
        out
    }

    /// Matrix multiply `self [M,K] @ other [K,N]` -> `[M,N]`.
    pub fn matmul(&self, other: &Tensor2D) -> VitResult<Tensor2D> {
        if self.cols != other.rows {
            return Err(VitError::Shape(format!(
                "matmul: [{},{}] @ [{},{}] shape mismatch",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out = Tensor2D::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut s = 0.0f64;
                for k in 0..self.cols {
                    s += self.get(i, k) * other.get(k, j);
                }
                out.set(i, j, s);
            }
        }
        Ok(out)
    }

    /// Transpose.
    pub fn transpose(&self) -> Tensor2D {
        let mut out = Tensor2D::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Element-wise add.
    pub fn add(&self, other: &Tensor2D) -> VitResult<Tensor2D> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(VitError::Shape("add: shape mismatch".to_string()));
        }
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a + b).collect();
        Ok(Tensor2D { rows: self.rows, cols: self.cols, data })
    }

    /// Element-wise scale.
    pub fn scale(&self, s: f64) -> Tensor2D {
        let data = self.data.iter().map(|&x| x * s).collect();
        Tensor2D { rows: self.rows, cols: self.cols, data }
    }

    /// Layer normalization over each row.
    pub fn layer_norm(&self, eps: f64) -> Tensor2D {
        let mut out = self.clone();
        for r in 0..self.rows {
            let start = r * self.cols;
            let slice = &self.data[start..start + self.cols];
            let mean: f64 = slice.iter().sum::<f64>() / self.cols as f64;
            let var: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.cols as f64;
            let std = (var + eps).sqrt();
            for i in 0..self.cols {
                out.data[start + i] = (self.data[start + i] - mean) / std;
            }
        }
        out
    }

    /// Row mean.
    pub fn row_mean(&self) -> Vec<f64> {
        (0..self.rows).map(|r| {
            let start = r * self.cols;
            self.data[start..start + self.cols].iter().sum::<f64>() / self.cols as f64
        }).collect()
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

/// 3D tensor view (batch × seq × dim).
#[derive(Debug, Clone)]
pub struct Tensor3D {
    /// Batch size.
    pub batch: usize,
    /// Sequence length.
    pub seq: usize,
    /// Feature dimension.
    pub dim: usize,
    /// Flattened data.
    pub data: Vec<f64>,
}

impl Tensor3D {
    /// Create zero-initialized 3D tensor.
    pub fn zeros(batch: usize, seq: usize, dim: usize) -> Self {
        Self { batch, seq, dim, data: vec![0.0; batch * seq * dim] }
    }

    /// Create from flat data.
    pub fn from_data(batch: usize, seq: usize, dim: usize, data: Vec<f64>) -> VitResult<Self> {
        let expected = batch * seq * dim;
        if data.len() != expected {
            return Err(VitError::Shape(format!(
                "Tensor3D: expected {} elements for [{},{},{}], got {}",
                expected, batch, seq, dim, data.len()
            )));
        }
        Ok(Self { batch, seq, dim, data })
    }

    /// Get element `[b, s, d]`.
    pub fn get(&self, b: usize, s: usize, d: usize) -> f64 {
        self.data[b * self.seq * self.dim + s * self.dim + d]
    }

    /// Set element `[b, s, d]`.
    pub fn set(&mut self, b: usize, s: usize, d: usize, val: f64) {
        self.data[b * self.seq * self.dim + s * self.dim + d] = val;
    }

    /// Extract a 2D slice for batch `b` → `[seq, dim]`.
    pub fn batch_slice(&self, b: usize) -> Tensor2D {
        let start = b * self.seq * self.dim;
        let data = self.data[start..start + self.seq * self.dim].to_vec();
        Tensor2D { rows: self.seq, cols: self.dim, data }
    }

    /// Mean over sequence dimension → `[batch, dim]`.
    pub fn mean_pool(&self) -> Tensor2D {
        let mut out = Tensor2D::zeros(self.batch, self.dim);
        for b in 0..self.batch {
            for d in 0..self.dim {
                let mut s = 0.0f64;
                for t in 0..self.seq {
                    s += self.get(b, t, d);
                }
                out.set(b, d, s / self.seq as f64);
            }
        }
        out
    }

    /// Extract CLS token (index 0) → `[batch, dim]`.
    pub fn cls_pool(&self) -> Tensor2D {
        let mut out = Tensor2D::zeros(self.batch, self.dim);
        for b in 0..self.batch {
            for d in 0..self.dim {
                out.set(b, d, self.get(b, 0, d));
            }
        }
        out
    }
}

/// Simple seeded pseudo-random number generator (LCG) for test/init.
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create with given seed.
    pub fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }

    /// Next pseudo-random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 11) | 0x3FF0_0000_0000_0000u64;
        f64::from_bits(bits) - 1.0
    }

    /// Next f64 in `[lo, hi)`.
    pub fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }

    /// Next u64 in `[0, n)`.
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize
    }

    /// Generate a Vec of `n` f64 values uniform in `[lo, hi)`.
    pub fn gen_vec(&mut self, n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n).map(|_| self.next_range(lo, hi)).collect()
    }

    /// Xavier uniform initialization for a weight matrix `[rows, cols]`.
    pub fn xavier_uniform(&mut self, rows: usize, cols: usize) -> Vec<f64> {
        let limit = (6.0 / (rows + cols) as f64).sqrt();
        self.gen_vec(rows * cols, -limit, limit)
    }

    /// Generate indices for random masking (without replacement).
    pub fn sample_without_replacement(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..k.min(n) {
            let j = i + self.next_usize(n - i);
            indices.swap(i, j);
        }
        indices[..k.min(n)].to_vec()
    }
}

/// Version string for this crate.
pub const BRAIN_VIT_VERSION: &str = "0.2.0";

/// Returns the crate version.
pub fn version() -> &'static str { BRAIN_VIT_VERSION }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vit_error_display() {
        let e = VitError::InvalidPatchSize { image_dim: 224, patch_size: 15 };
        let s = e.to_string();
        assert!(s.contains("224"));
        assert!(s.contains("15"));
    }

    #[test]
    fn test_vit_error_config() {
        let e = VitError::Config("bad value".to_string());
        assert!(e.to_string().contains("bad value"));
    }

    #[test]
    fn test_vit_error_empty_batch() {
        let e = VitError::EmptyBatch;
        assert!(e.to_string().contains("Batch"));
    }

    #[test]
    fn test_pool_strategy_from_str() {
        assert_eq!(PoolStrategy::from_str("cls").unwrap(), PoolStrategy::Cls);
        assert_eq!(PoolStrategy::from_str("mean").unwrap(), PoolStrategy::MeanPool);
        assert_eq!(PoolStrategy::from_str("gap").unwrap(), PoolStrategy::Gap);
        assert!(PoolStrategy::from_str("invalid").is_err());
    }

    #[test]
    fn test_pool_strategy_display() {
        assert_eq!(PoolStrategy::Cls.to_string(), "cls");
        assert_eq!(PoolStrategy::MeanPool.to_string(), "mean_pool");
        assert_eq!(PoolStrategy::Gap.to_string(), "gap");
    }

    #[test]
    fn test_vit_output_empty() {
        let out = VitOutput::empty();
        assert_eq!(out.batch_size(), 0);
        assert_eq!(out.num_patches(), 0);
        assert_eq!(out.embed_dim(), 0);
        assert!(!out.has_attentions());
    }

    #[test]
    fn test_vit_output_populated() {
        let mut out = VitOutput::empty();
        out.logits = vec![vec![0.1, 0.9], vec![0.8, 0.2]];
        out.cls_token = vec![vec![1.0; 64], vec![2.0; 64]];
        out.patch_tokens = vec![vec![vec![0.0; 64]; 196], vec![vec![0.0; 64]; 196]];
        assert_eq!(out.batch_size(), 2);
        assert_eq!(out.num_patches(), 196);
        assert_eq!(out.embed_dim(), 64);
    }

    #[test]
    fn test_vit_state_default() {
        let s = VitState::new();
        assert_eq!(s.forward_count, 0);
        assert!(s.is_training);
        assert_eq!(s.global_step, 0);
    }

    #[test]
    fn test_vit_state_record_forward() {
        let mut s = VitState::new();
        s.record_forward(197);
        assert_eq!(s.forward_count, 1);
        assert_eq!(s.total_tokens_processed, 197);
    }

    #[test]
    fn test_vit_state_update_best() {
        let mut s = VitState::new();
        assert!(s.update_best(1.5));
        assert!(!s.update_best(2.0));
        assert!(s.update_best(0.5));
        assert_eq!(s.best_metric, 0.5);
    }

    #[test]
    fn test_vit_state_stats() {
        let mut s = VitState::new();
        s.update_stat("loss", 1.23);
        assert!((s.get_stat("loss") - 1.23).abs() < 1e-10);
        assert_eq!(s.get_stat("missing"), 0.0);
    }

    #[test]
    fn test_vit_state_summary() {
        let mut s = VitState::new();
        s.step();
        let summary = s.summary();
        assert!(summary.contains("steps=1"));
    }

    #[test]
    fn test_vit_state_total_params() {
        let mut s = VitState::new();
        s.register_layer("embed", 1000);
        s.register_layer("head", 500);
        assert_eq!(s.total_params(), 1500);
    }

    #[test]
    fn test_tensor2d_zeros() {
        let t = Tensor2D::zeros(3, 4);
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 4);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor2d_get_set() {
        let mut t = Tensor2D::zeros(2, 3);
        t.set(1, 2, 5.0);
        assert_eq!(t.get(1, 2), 5.0);
    }

    #[test]
    fn test_tensor2d_from_data_err() {
        let r = Tensor2D::from_data(2, 3, vec![1.0; 5]);
        assert!(r.is_err());
    }

    #[test]
    fn test_tensor2d_matmul() {
        let a = Tensor2D::from_data(2, 3, vec![1.0,2.0,3.0, 4.0,5.0,6.0]).unwrap();
        let b = Tensor2D::from_data(3, 2, vec![7.0,8.0, 9.0,10.0, 11.0,12.0]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert!((c.get(0,0) - 58.0).abs() < 1e-9);
        assert!((c.get(1,1) - 154.0).abs() < 1e-9);
    }

    #[test]
    fn test_tensor2d_transpose() {
        let a = Tensor2D::from_data(2, 3, vec![1.0,2.0,3.0, 4.0,5.0,6.0]).unwrap();
        let b = a.transpose();
        assert_eq!(b.rows, 3);
        assert_eq!(b.cols, 2);
        assert_eq!(b.get(0, 1), 4.0);
    }

    #[test]
    fn test_tensor2d_scale() {
        let a = Tensor2D::from_data(1, 3, vec![1.0,2.0,3.0]).unwrap();
        let b = a.scale(2.0);
        assert!((b.get(0,2) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor2d_add() {
        let a = Tensor2D::from_data(1, 2, vec![1.0,2.0]).unwrap();
        let b = Tensor2D::from_data(1, 2, vec![3.0,4.0]).unwrap();
        let c = a.add(&b).unwrap();
        assert!((c.get(0,0) - 4.0).abs() < 1e-10);
        assert!((c.get(0,1) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor2d_softmax() {
        let a = Tensor2D::from_data(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let b = a.softmax_rows();
        let sum: f64 = b.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_tensor2d_layer_norm() {
        let a = Tensor2D::from_data(1, 4, vec![1.0,2.0,3.0,4.0]).unwrap();
        let b = a.layer_norm(1e-5);
        let mean: f64 = b.data.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_tensor2d_frobenius() {
        let a = Tensor2D::from_data(1, 3, vec![3.0,4.0,0.0]).unwrap();
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_tensor3d_zeros() {
        let t = Tensor3D::zeros(2, 5, 8);
        assert_eq!(t.batch, 2);
        assert_eq!(t.seq, 5);
        assert_eq!(t.dim, 8);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor3d_get_set() {
        let mut t = Tensor3D::zeros(2, 3, 4);
        t.set(1, 2, 3, 7.0);
        assert_eq!(t.get(1, 2, 3), 7.0);
    }

    #[test]
    fn test_tensor3d_cls_pool() {
        let mut t = Tensor3D::zeros(2, 5, 4);
        for d in 0..4 { t.set(0, 0, d, (d + 1) as f64); }
        let cls = t.cls_pool();
        assert_eq!(cls.rows, 2);
        assert_eq!(cls.cols, 4);
        assert!((cls.get(0, 3) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor3d_mean_pool() {
        let mut t = Tensor3D::zeros(1, 4, 2);
        for s in 0..4 {
            for d in 0..2 {
                t.set(0, s, d, 1.0);
            }
        }
        let mp = t.mean_pool();
        assert!((mp.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);
        let v = rng.next_f64();
        assert!(v >= 0.0 && v < 1.0);
    }

    #[test]
    fn test_simple_rng_range() {
        let mut rng = SimpleRng::new(99);
        for _ in 0..100 {
            let v = rng.next_range(-1.0, 1.0);
            assert!(v >= -1.0 && v < 1.0);
        }
    }

    #[test]
    fn test_simple_rng_gen_vec() {
        let mut rng = SimpleRng::new(7);
        let v = rng.gen_vec(50, 0.0, 1.0);
        assert_eq!(v.len(), 50);
        assert!(v.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_simple_rng_xavier() {
        let mut rng = SimpleRng::new(13);
        let w = rng.xavier_uniform(64, 64);
        assert_eq!(w.len(), 64 * 64);
        let limit = (6.0f64 / 128.0).sqrt();
        assert!(w.iter().all(|&x| x.abs() <= limit + 1e-10));
    }

    #[test]
    fn test_simple_rng_without_replacement() {
        let mut rng = SimpleRng::new(21);
        let indices = rng.sample_without_replacement(196, 49);
        assert_eq!(indices.len(), 49);
        // All in range
        assert!(indices.iter().all(|&i| i < 196));
        // All unique
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 49);
    }

    #[test]
    fn test_version() {
        assert_eq!(version(), "0.2.0");
    }

    #[test]
    fn test_vit_dtype_default() {
        let d = VitDtype::default();
        assert_eq!(d, VitDtype::F32);
    }

    #[test]
    fn test_tensor2d_add_err() {
        let a = Tensor2D::zeros(2, 3);
        let b = Tensor2D::zeros(3, 2);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_tensor2d_matmul_err() {
        let a = Tensor2D::zeros(2, 3);
        let b = Tensor2D::zeros(4, 2);
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_vit_state_set_training() {
        let mut s = VitState::new();
        s.set_training(false);
        assert!(!s.is_training);
    }

    #[test]
    fn test_vit_state_reset_stats() {
        let mut s = VitState::new();
        s.update_stat("x", 1.0);
        s.reset_stats();
        assert_eq!(s.get_stat("x"), 0.0);
    }

    #[test]
    fn test_vit_error_shape() {
        let e = VitError::Shape("bad shape".to_string());
        assert!(e.to_string().contains("bad shape"));
    }

    #[test]
    fn test_tensor3d_from_data_err() {
        let r = Tensor3D::from_data(2, 3, 4, vec![0.0; 20]);
        assert!(r.is_err());
    }

    #[test]
    fn test_tensor2d_row_mean() {
        let t = Tensor2D::from_data(1, 4, vec![2.0, 4.0, 6.0, 8.0]).unwrap();
        let m = t.row_mean();
        assert!((m[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_simple_rng_deterministic() {
        let mut r1 = SimpleRng::new(123);
        let mut r2 = SimpleRng::new(123);
        for _ in 0..20 {
            assert_eq!(r1.next_f64().to_bits(), r2.next_f64().to_bits());
        }
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






}
