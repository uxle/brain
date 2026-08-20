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
            VitError::InvalidPatchSize {
                image_dim,
                patch_size,
            } => write!(
                f,
                "Image dim {} not divisible by patch size {}",
                image_dim, patch_size
            ),
            VitError::ResolutionMismatch { expected, got } => {
                write!(f, "Resolution mismatch: expected {}, got {}", expected, got)
            }
            VitError::DimMismatch { expected, got } => {
                write!(f, "Dim mismatch: expected {}, got {}", expected, got)
            }
            VitError::EmptyBatch => write!(f, "Batch size must be > 0"),
            VitError::Config(msg) => write!(f, "Config error: {}", msg),
            VitError::Checkpoint(msg) => write!(f, "Checkpoint error: {}", msg),
            VitError::Shape(msg) => write!(f, "Shape error: {}", msg),
            VitError::Numerical(msg) => write!(f, "Numerical error: {}", msg),
            VitError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
            VitError::Io(msg) => write!(f, "I/O error: {}", msg),
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
            other => Err(VitError::Config(format!(
                "Unknown pool strategy: {}",
                other
            ))),
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
    pub fn new() -> Self {
        Self::default()
    }

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
    pub fn set_training(&mut self, training: bool) {
        self.is_training = training;
    }

    /// Increment global step.
    pub fn step(&mut self) {
        self.global_step += 1;
    }

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
    pub fn reset_stats(&mut self) {
        self.stats.clear();
    }

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
            self.global_step,
            self.forward_count,
            self.total_tokens_processed,
            self.is_training,
            self.best_metric
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
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create from data, verifying shape.
    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> VitResult<Self> {
        if data.len() != rows * cols {
            return Err(VitError::Shape(format!(
                "Expected {} elements for shape [{}, {}], got {}",
                rows * cols,
                rows,
                cols,
                data.len()
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
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Ok(Tensor2D {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    /// Element-wise scale.
    pub fn scale(&self, s: f64) -> Tensor2D {
        let data = self.data.iter().map(|&x| x * s).collect();
        Tensor2D {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Layer normalization over each row.
    pub fn layer_norm(&self, eps: f64) -> Tensor2D {
        let mut out = self.clone();
        for r in 0..self.rows {
            let start = r * self.cols;
            let slice = &self.data[start..start + self.cols];
            let mean: f64 = slice.iter().sum::<f64>() / self.cols as f64;
            let var: f64 =
                slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.cols as f64;
            let std = (var + eps).sqrt();
            for i in 0..self.cols {
                out.data[start + i] = (self.data[start + i] - mean) / std;
            }
        }
        out
    }

    /// Row mean.
    pub fn row_mean(&self) -> Vec<f64> {
        (0..self.rows)
            .map(|r| {
                let start = r * self.cols;
                self.data[start..start + self.cols].iter().sum::<f64>() / self.cols as f64
            })
            .collect()
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
        Self {
            batch,
            seq,
            dim,
            data: vec![0.0; batch * seq * dim],
        }
    }

    /// Create from flat data.
    pub fn from_data(batch: usize, seq: usize, dim: usize, data: Vec<f64>) -> VitResult<Self> {
        let expected = batch * seq * dim;
        if data.len() != expected {
            return Err(VitError::Shape(format!(
                "Tensor3D: expected {} elements for [{},{},{}], got {}",
                expected,
                batch,
                seq,
                dim,
                data.len()
            )));
        }
        Ok(Self {
            batch,
            seq,
            dim,
            data,
        })
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
        Tensor2D {
            rows: self.seq,
            cols: self.dim,
            data,
        }
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
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Next pseudo-random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
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
pub fn version() -> &'static str {
    BRAIN_VIT_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vit_error_display() {
        let e = VitError::InvalidPatchSize {
            image_dim: 224,
            patch_size: 15,
        };
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
        assert_eq!(
            PoolStrategy::from_str("mean").unwrap(),
            PoolStrategy::MeanPool
        );
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
        let a = Tensor2D::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor2D::from_data(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert!((c.get(0, 0) - 58.0).abs() < 1e-9);
        assert!((c.get(1, 1) - 154.0).abs() < 1e-9);
    }

    #[test]
    fn test_tensor2d_transpose() {
        let a = Tensor2D::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = a.transpose();
        assert_eq!(b.rows, 3);
        assert_eq!(b.cols, 2);
        assert_eq!(b.get(0, 1), 4.0);
    }

    #[test]
    fn test_tensor2d_scale() {
        let a = Tensor2D::from_data(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let b = a.scale(2.0);
        assert!((b.get(0, 2) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor2d_add() {
        let a = Tensor2D::from_data(1, 2, vec![1.0, 2.0]).unwrap();
        let b = Tensor2D::from_data(1, 2, vec![3.0, 4.0]).unwrap();
        let c = a.add(&b).unwrap();
        assert!((c.get(0, 0) - 4.0).abs() < 1e-10);
        assert!((c.get(0, 1) - 6.0).abs() < 1e-10);
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
        let a = Tensor2D::from_data(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = a.layer_norm(1e-5);
        let mean: f64 = b.data.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_tensor2d_frobenius() {
        let a = Tensor2D::from_data(1, 3, vec![3.0, 4.0, 0.0]).unwrap();
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
        for d in 0..4 {
            t.set(0, 0, d, (d + 1) as f64);
        }
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
