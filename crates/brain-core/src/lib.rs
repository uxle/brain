//! Brain Core - A pure-Rust deep learning framework.
//!
//! This crate provides fundamental building blocks for deep learning computations,
//! including tensor operations, linear algebra, random number generation,
//! memory management, and serialization.
//!
//! # Version
//!
//! Current version: 0.1.0
//!
//! # Feature Comparison
//!
//! | Feature          | Brain Core | PyTorch | TensorFlow |
//! |------------------|-----------|---------|------------|
//! | Pure Rust        | Yes       | No (C++) | No (C++)   |
//! | No dependencies | Yes       | Heavy    | Heavy      |
//! | GPU support     | Planned   | Yes      | Yes        |
//! | Auto-diff       | Planned   | Yes      | Yes        |
//! | Format support  | Bin/JSON  | Many     | Many      |
//!
//! # Architecture
//!
//! Brain Core is designed as a foundation library with no external dependencies.
//! All computations are performed in pure Rust using the CPU backend.
//!
//! ## Module Organization
//!
//! ```text
//! brain-core/
//! ├── src/
//! │   ├── lib.rs          # Crate root, re-exports, prelude
//! │   ├── error.rs        # Error types and macros
//! │   ├── dtype.rs        # Data type definitions
//!   │   ├── device.rs       # Device abstraction
//!   │   ├── shape.rs        # Shape manipulation
//!   ├── tensor/
//! │   │   ├── mod.rs        # Tensor module root
//!   │   ├── impl.rs       # Tensor struct and core ops
//!   │   ├── arithmetic.rs  # Arithmetic operations
//!   │   ├── math.rs        # Math functions
//!   │   ├── linalg.rs      # Linear algebra
//!   │   ├── reduction.rs  # Reduction operations
//!   │   └── indexing.rs    # Indexing operations
//!   ├── memory.rs        # Memory management
//!   ├── random.rs        # Random number generation
//!   └── serialization.rs  # Save/load tensors
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use brain_core::prelude::*;
//!
//! // Create tensors
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
//! let b = Tensor::identity(3);
//!
//! // Operations
//! let c = tensor::arithmetic::matmul(&a, &b);
//! let d = tensor::math::sigmoid(&c);
//!
//! // Linear algebra
//! let det = tensor::linalg::det(&a);
//! let inv = tensor::linalg::inv(&a);
//!
//! // Statistics
//! let stats = c.statistics();
//! ```
//!
//! # Design Principles
//!
//! 1. **Zero dependencies**: Works with just the Rust standard library
//! 2. **Correct over fast**: All implementations prioritize correctness
//! 3. **API ergonomics**: Mirrors NumPy/PyTorch conventions where possible
//! 4. **Safety**: Rust's type system catches many errors at compile time
//!
//! # Performance Notes
//!
//! Since this is a pure-Rust implementation without SIMD or BLAS, performance
//! is limited compared to frameworks that use optimized backends. For production
//! use, consider adding a BLAS or SIMD backend.
//!
//! The framework is designed so that compute backends can be swapped in by
//! implementing trait interfaces without changing the user-facing API.

pub const VERSION: &str = "0.1.0";
pub const GIT_HASH: &str = "dev";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 1;
pub const PATCH_VERSION: u32 = 0;

// Re-export all modules
pub mod error;
pub mod dtype;
pub mod device;
pub mod shape;
pub mod tensor;
pub mod memory;
pub mod random;
pub mod serialization;

// =============================================================================
// Convenience Re-exports
// =============================================================================

pub use error::{BrainError, BrainResult};
pub use dtype::DType;
pub use device::Device;
pub use shape::Shape;
pub use tensor::{Tensor, TensorStats, TensorIter, TensorIterMut, TensorIndex, Layout, TensorFlags};
pub use tensor::arithmetic;
pub use tensor::math;
pub use tensor::linalg;
pub use tensor::reduction;
pub use tensor::indexing;

// =============================================================================
// Prelude Module
// =============================================================================

/// Common imports for convenient framework usage.
///
/// ```ignore
//! use brain_core::prelude::*;
//! ```
pub mod prelude {
    pub use crate::error::{BrainError, BrainResult};
    pub use crate::dtype::DType;
    pub use crate::device::Device;
    pub use crate::shape::Shape;
    pub use crate::tensor::Tensor;
    pub use crate::VERSION;
}

// =============================================================================
// Build Information
// =============================================================================

/// Returns compile-time Rustc version info.
pub fn rustc_version() -> &'static str {
    env!("RUSTC_VERSION")
}

/// Returns the target architecture.
pub fn target_arch() -> &'static str {
    env!("CFG_TARGET_ARCH")
}

/// Returns the target operating system.
pub fn target_os() -> &'static str {
    env!("CFG_OS")
}

/// Returns the complete version string.
pub fn version() -> &'static str {
    VERSION
}

/// Returns the git hash placeholder.
pub fn git_hash() -> &'static str {
    GIT_HASH
}

/// Returns a formatted full version string.
pub fn version_string() -> String {
    format!("brain-core v{} ({} {})", VERSION, GIT_HASH, target_arch(), target_os())
}

/// Returns build configuration information.
pub fn build_info() -> String {
    format!(
        "brain-core v{} | Rust {} | {} | {}",
        VERSION,
        rustc_version(),
        target_arch(),
        target_os(),
    )
}

/// Returns the number of modules in the framework.
pub fn module_count() -> usize {
    9 // error, dtype, device, shape, tensor, memory, random, serialization, lib
}

// =============================================================================
// Framework Initialization
// =============================================================================

/// Global initialization state.
static INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Initializes the Brain framework. Must be called before using any framework function.
pub fn initialize() {
    INITIALIZED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Returns true if the framework has been initialized.
pub fn is_initialized() -> bool {
    INITIALIZED.load(std::sync::atomic::Ordering::Relaxed)
}

// =============================================================================
// Configuration
// =============================================================================

/// Global framework configuration options.
#[derive(Debug, Clone)]
pub struct Config {
    /// Whether to print debug information.
    pub debug: bool,
    /// Default device for new tensors.
    pub default_device: Device,
    /// Default data type for new tensors.
    pub default_dtype: DType,
    /// Seed for the global RNG.
    pub seed: u64,
    /// Maximum number of threads for parallel operations.
    pub max_threads: usize,
    /// Memory limit in bytes (0 = no limit).
    pub memory_limit: usize,
    /// Whether to enable gradient checking.
    pub grad_checking: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            debug: false,
            default_device: Device::Cpu,
            default_dtype: DType::F64,
            seed: 42,
            max_threads: 1,
            memory_limit: 0,
            grad_checking: false,
        }
    }
}

impl Config {
    /// Creates a new configuration with sensible defaults.
    pub fn new() -> Self { Self::default() }

    /// Sets the debug flag.
    pub fn with_debug(mut self, debug: bool) -> Self { self.debug = debug; self }

    /// Sets the default device.
    pub fn with_device(mut self, device: Device) -> Self { self.default_device = device; self }

    /// Sets the default data type.
    pub fn with_dtype(mut self, dtype: DType) -> Self { self.default_dtype = dtype; self }

    /// Sets the global seed.
    pub fn with_seed(mut self, seed: u64) -> Self { self.seed = seed; self }

    /// Sets the maximum thread count.
    pub fn with_max_threads(mut self, max_threads: usize) -> Self { self.max_threads = max_threads; self }

    /// Sets the memory limit in bytes.
    pub fn with_memory_limit(mut self, limit: usize) -> Self { self.memory_limit = limit; self }

    /// Enables gradient checking.
    pub fn with_grad_checking(mut self, enabled: bool) -> Self { self.grad_checking = enabled; self }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_threads == 0 {
            return Err("max_threads must be at least 1".into());
        }
        Ok(())
    }

    /// Returns a summary of the configuration.
    pub fn summary(&self) -> String {
        format!(
            "Config {{\n  debug: {},\n  default_device: {},\n  default_dtype: {},\n  seed: {},\n  max_threads: {},\n  memory_limit: {},\n  grad_checking: {},\n}}",
            self.debug, self.default_device, self.default_dtype, self.seed,
            self.max_threads,
            if self.memory_limit == 0 { "unlimited".into() } else { format!("{} bytes", self.memory_limit) },
            self.grad_checking,
        )
    }
}

/// Global configuration instance.
static CONFIG: std::sync::RwLock<Config> = std::sync::RwLock::new(Config::default());

/// Gets a copy of the global configuration.
pub fn config() -> Config {
    CONFIG.read().unwrap().clone()
}

/// Sets the global configuration.
pub fn set_config(config: Config) -> Result<(), String> {
    config.validate()?;
    *CONFIG.write().unwrap() = config;
    Ok(())
}

/// Updates the global configuration using a closure.
pub fn with_config<F: FnOnce(&mut Config) -> T, T>(f: F) -> T {
    let mut config = CONFIG.write().unwrap();
    let result = f(&mut config);
    result
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Returns the framework version as a tuple.
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Checks if a version string is compatible with the current version.
pub fn check_version_compat(version_str: &str) -> Result<(), String> {
    let parts: Vec<&str> = version_str.split('.').collect();
    if parts.len() != 3 {
        return Err(format!("Invalid version string: {}", version_str));
    }
    let major: u32 = parts[0].parse().unwrap_or(0);
    let minor: u32 = parts[1].parse().unwrap_or(0);
    let _patch: u32 = parts[2].parse().unwrap_or(0);
    if major > MAJOR_VERSION {
        return Err(format!(
            "Version {}.{}.{} is not compatible with {}",
            major, minor, _patch, VERSION
        ));
    }
    Ok(())
}

/// Returns the total number of source lines in the framework (approximate).
pub fn source_lines() -> usize {
    let mut total = 0;
    let _ = &mut total;
    // This would count actual source lines at compile time
    // For now return a placeholder
    15000
}

/// Returns approximate memory usage of the framework code segment (code + data).
pub fn framework_size_bytes() -> usize {
    // Approximate code size
    500000 // ~500KB for the framework binary
}

/// Returns a formatted string describing all loaded modules.
pub fn module_info() -> String {
    let mut info = String::new();
    info.push_str("Loaded modules:\n");
    info.push_str("  error.rs    - Error types and handling\n");
    info.push_str("  dtype.rs    - Data type definitions\n");
    info.push_str("  device.rs   - Device abstraction\n");
    info.push_str("  shape.rs    - Shape manipulation\n");
    info.push_str("  tensor/      - Tensor operations:\n");
    info.push_str("    mod.rs      - Module root and types\n");
    info.push_str("    impl.rs      - Core Tensor struct\n");
    info.push_str("    arithmetic - Arithmetic operations\n");
    info.push_str("    math        - Mathematical functions\n");
    info.push_str("    linalg      - Linear algebra\n");
    info.push_str("    reduction   - Reduction operations\n");
    info.push_str("    indexing    - Indexing operations\n");
    info.push_str("  memory.rs   - Memory management\n");
    info.push_str("  random.rs   - Random number generation\n");
    info.push_str("  serialization.rs - Save/load tensors\n");
    info.push_str("  lib.rs      - Crate root\n");
    info
}

// =============================================================================
// Backward Compatibility Helpers
// =============================================================================

/// Warns about deprecated features (placeholder).
pub fn deprecation_warn(feature: &str) {
    eprintln!("Warning: '{}' is deprecated and may be removed in a future version", feature);
}

/// Logs a debug message if debug mode is enabled.
pub fn debug_log(msg: &str) {
    if config().debug {
        eprintln!("[BRAIN-DEBUG] {}", msg);
    }
}

/// Logs a performance warning if a slow path is taken.
pub fn perf_warn(msg: &str) {
    eprintln!("[BRAIN-PERF] {}", msg);
}

// =============================================================================
// Type Aliases for Common Types
// =============================================================================

/// A type alias for a reference to a tensor slice.
pub type TensorSlice<'a> = &'a [f64];

/// A type alias for a mutable tensor slice.
pub type TensorSliceMut<'a> = &'a mut [f64];

/// A type alias for a borrowed 1D tensor (vector).
pub type TensorView<'a> = &'a Tensor;

/// A type alias for a mutable borrowed tensor.
pub type TensorViewMut<'a> = &'a mut Tensor;

/// A type alias for an owned tensor (for use in collections).
pub type OwnedTensor = Tensor;

/// A type alias for boxed tensors.
pub type BoxTensor = Box<Tensor>;

/// A type alias for thread-safe tensor references.
pub type ArcTensor = std::sync::Arc<Tensor>;

/// A type alias for a 1D tensor (vector).
pub type Vector = Tensor;

/// A type alias for a 2D tensor (matrix).
pub type Matrix = Tensor;

/// A type alias for a 3D tensor.
pub type Tensor3D = Tensor;

/// A type alias for a 4D tensor (batch tensor).
pub type BatchTensor = Tensor;

/// A type alias for a scalar tensor.
pub type Scalar = Tensor;

/// A type alias for a gradient tensor (tensor with requires_grad).
pub type GradTensor = Tensor;

/// A type alias for a parameter tensor.
pub type Parameter = Tensor;

/// A type alias for a bias vector.
pub type Bias = Tensor;

/// A type alias for a weight matrix.
pub type Weights = Tensor;

/// A type alias for an activation output.
pub type Activation = Tensor;

/// A type alias for logits output.
pub type Logits = Tensor;

/// A type alias for a loss value.
pub type Loss = f64;

/// A type alias for gradients.
pub type Gradients = Vec<Tensor>;

/// A type alias for a layer of parameters.
pub type LayerParams = Vec<Tensor>;

/// A type alias for a batch of tensors.
pub type Batch = Vec<Tensor>;

/// A type alias for model outputs.
pub type ModelOutput = Vec<Tensor>;

/// A type alias for a model state.
pub type ModelState = Vec<Tensor>;

/// A type alias for a sequence of tensors.
pub type Sequence = Vec<Tensor>;

/// A type alias for hidden states in RNNs.
pub type HiddenState = Tensor;

/// A type alias for attention weights.
pub type AttentionWeights = Tensor;

/// A type alias for positional encodings.
pub type PositionalEncoding = Tensor;

// =============================================================================
// Comparison and Equality Helpers
// =============================================================================

/// Compares two tensors element-wise with a tolerance.
pub fn tensors_close(a: &Tensor, b: &Tensor, atol: f64, rtol: f64) -> bool {
    if a.shape() != b.shape() { return false; }
    for (av, bv) in a.data().iter().zip(b.data().iter()) {
        let diff = (av - bv).abs();
        if diff > atol && diff > rtol * bv.abs().max(atol) {
            return false;
        }
    }
    true
}

/// Asserts that two tensors are approximately equal.
pub fn assert_close(a: &Tensor, b: &Tensor, atol: f64, rtol: f64) {
    if !tensors_close(a, b, atol, rtol) {
        panic!(
            "Tensors are not close:\n  a: {:?}\n  b: {:?}\n  atol={}, rtol={}",
            a, b, atol, rtol
        );
    }
}

/// Returns the number of elements that differ between two tensors.
pub fn count_differences(a: &Tensor, b: &Tensor) -> usize {
    if a.shape() != b.shape() { return usize::MAX; }
    a.data().iter().zip(b.data().iter()).filter(|(a, b)| (a - b).abs() > 1e-10).count()
}

/// Returns the maximum absolute difference between two tensors.
pub fn max_difference(a: &Tensor, b: &Tensor) -> f64 {
    if a.shape() != b.shape() { return f64::NAN; }
    a.data().iter().zip(b.data().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(f64::NEG_INFINITY, f64::max)
}

// =============================================================================
// Tensor Creation Helpers
// =============================================================================

/// Creates a 1D tensor (vector) from values.
pub fn vec_from_values(values: &[f64]) -> Tensor {
    Tensor::from_slice(values, vec![values.len()])
}

/// Creates a 2D tensor (matrix) from values in row-major order.
pub fn matrix_from_values(values: &[f64], rows: usize, cols: usize) -> Tensor {
    assert_eq!(values.len(), rows * cols);
    Tensor::new(values.to_vec(), vec![rows, cols])
}

/// Creates a diagonal matrix from values.
pub fn diag_from_values(values: &[f64]) -> Tensor {
    crate::tensor::Tensor::from_diag(values)
}

/// Creates an upper triangular matrix from values.
pub fn triu_from_values(values: &[f64], n: usize) -> Tensor {
    let mut data = vec![0.0; n * n];
    let mut idx = 0;
    for i in 0..n {
        for j in i..n {
            if idx < values.len() { data[i * n + j] = values[idx]; }
            idx += 1;
        }
    }
    Tensor::new(data, vec![n, n])
}

/// Creates a lower triangular matrix from values.
pub fn tril_from_values(values: &[f64], n: usize) -> Tensor {
    let mut data = vec![0.0; n * n];
    let mut idx = 0;
    for i in 0..n {
        for j in 0..=i {
            if idx < values.len() { data[i * n + j] = values[idx]; }
            idx += 1;
        }
    }
    Tensor::new(data, vec![n, n])
}

/// Creates a symmetric matrix from upper triangle values.
pub fn symmetric_from_values(values: &[f64], n: usize) -> Tensor {
    let mut data = vec![0.0; n * n];
    let mut idx = 0;
    for i in 0..n {
        for j in i..n {
            if idx < values.len() {
                let v = values[idx];
                data[i * n + j] = v;
                data[j * n + i] = v;
            }
            idx += 1;
        }
    }
    Tensor::new(data, vec![n, n])
}

/// Creates a tensor filled with random normal values.
pub fn randn(shape: Vec<usize>, mean: f64, std: f64) -> Tensor {
    let numel: usize = shape.iter().product();
    let mut rng = crate::random::default_rng();
    let mut data = vec![0.0; numel];
    rng.fill_normal(&mut data, mean, std);
    Tensor::new(data, shape)
}

/// Creates a tensor filled with random uniform values in [0, 1).
pub fn randu(shape: Vec<usize>) -> Tensor {
    let numel: usize = shape.iter().product();
    let mut rng = crate::random::default_rng();
    let mut data = vec![0.0; numel];
    rng.fill_uniform(&mut data, 0.0, 1.0);
    Tensor::new(data, shape)
}

/// Creates a tensor filled with zeros like PyTorch's torch.zeros().
pub fn zeros_like(shape: Vec<usize>) -> Tensor {
    Tensor::zeros(shape)
}

/// Creates a tensor filled with ones like PyTorch's torch.ones().
pub fn ones_like(shape: Vec<usize>) -> Tensor {
    Tensor::ones(shape)
}

/// Creates a tensor filled with a specific value like PyTorch's torch.full().
pub fn full_like(shape: Vec<usize>, value: f64) -> Tensor {
    Tensor::full(shape, value)
}

/// Creates an identity matrix like numpy.eye().
pub fn eye_like(n: usize) -> Tensor {
    Tensor::identity(n)
}

/// Creates a random matrix with Kaiming initialization.
pub fn kaiming_matrix(rows: usize, cols: usize) -> Tensor {
    let fan_in = cols;
    let mut rng = crate::random::default_rng();
    let bound = (6.0 / fan_in as f64).sqrt();
    let mut data = vec![0.0; rows * cols];
    for v in data.iter_mut() { *v = rng.uniform(-bound, bound); }
    Tensor::new(data, vec![rows, cols])
}

/// Creates a random matrix with Xavier initialization.
pub fn xavier_matrix(rows: usize, cols: usize) -> Tensor {
    let fan_in = cols;
    let fan_out = rows;
    let mut rng = crate::random::default_rng();
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let mut data = vec![0.0; rows * cols];
    for v in data.iter_mut() { *v = rng.uniform(-bound, bound); }
    Tensor::new(data, vec![rows, cols])
}

/// Creates a batch of random matrices with Kaiming initialization.
pub fn kaiming_batch(batch: usize, rows: usize, cols: usize) -> Tensor {
    let fan_in = cols;
    let bound = (6.0 / fan_in as f64).sqrt();
    let numel = batch * rows * cols;
    let mut rng = crate::random::default_rng();
    let mut data = vec![0.0; numel];
    for v in data.iter_mut() { *v = rng.uniform(-bound, bound); }
    Tensor::new(data, vec![batch, rows, cols])
}

/// Creates a batch of random matrices with Xavier initialization.
pub fn xavier_batch(batch: usize, rows: usize, cols: usize) -> Tensor {
    let fan_in = cols;
    let fan_out = rows;
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let numel = batch * rows * cols;
    let mut rng = crate::random::default_rng();
    let mut data = vec![0.0; numel];
    for v in data.iter_mut() { *v = rng.uniform(-bound, bound); }
    Tensor::new(data, vec![batch, rows, cols])
}

// =============================================================================
// Common Shape Patterns
// =============================================================================

/// Creates a shape for an MLP with given layer sizes.
pub fn mlp_shape(layer_sizes: &[usize]) -> Vec<Vec<usize>> {
    let mut shapes = Vec::with_capacity(layer_sizes.len());
    for i in 0..layer_sizes.len() - 1 {
        shapes.push(vec![layer_sizes[i], layer_sizes[i + 1]]);
    }
    shapes
}

/// Returns the number of parameters for a sequence of layer shapes.
pub fn count_parameters(shapes: &[Vec<usize>]) -> usize {
    shapes.iter().map(|s| s.iter().product()).sum()
}

/// Computes the number of MACs for a given layer configuration.
pub fn compute_macs(shapes: &[Vec<usize>]) -> usize {
    let mut total = 0;
    for i in 0..shapes.len() - 1 {
        let in_features = shapes[i].iter().product();
        let out_features = shapes[i + 1].iter().product();
        total += in_features * out_features;
    }
    total
}

/// Computes the number of FLOPs for a given layer configuration.
pub fn compute_flops(shapes: &[Vec<usize>], flops_per_mac: usize) -> usize {
    compute_macs(shapes) * flops_per_mac
}

/// Creates a standard CNN output shape from input shape, kernel size, stride, and padding.
pub fn conv_output(input: &[usize], kernel: &[usize], stride: &[usize], padding: &[usize]) -> Vec<usize> {
    let n = input.get(0).copied().unwrap_or(1);
    let c = input.get(1).copied().unwrap_or(1);
    let h = input.get(2).copied().unwrap_or(1);
    let w = input.get(3).copied().unwrap_or(1);
    let kh = kernel.get(0).copied().unwrap_or(1);
    let kw = kernel.get(1).copied().unwrap_or(1);
    let sh = stride.get(0).copied().unwrap_or(1);
    let sw = stride.get(1).copied().unwrap_or(1);
    let ph = padding.get(0).copied().unwrap_or(0);
    let pw = padding.get(1).copied().unwrap_or(0);
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (w + 2 * pw - kw) / sw + 1;
    vec![n, c, oh, ow]
}

/// Creates a standard linear layer output shape from input and output features.
pub fn linear_output(input: &[usize], out_features: usize) -> Vec<usize> {
    let n = input.get(0).copied().unwrap_or(1);
    let rest: Vec<usize> = input.iter().skip(1).cloned().collect();
    let mut result = vec![n];
    result.extend(rest);
    result.push(out_features);
    result
}

// =============================================================================
// Validation Utilities
// =============================================================================

/// Validates that two shapes are equal, returning an error message if not.
pub fn require_same_shape(a: &[usize], b: &[usize], context: &str) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!("Shape mismatch in {}: expected {:?}, got {:?}", context, a, b))
    }
}

/// Validates that shape dimensions are positive.
pub fn require_positive_dims(shape: &[usize], context: &str) -> Result<(), String> {
    for (i, &dim) in shape.iter().enumerate() {
        if *dim == 0 {
            return Err(format!("Dimension {} must be positive in {}", i, context));
        }
    }
    Ok(())
}

/// Validates that the product of shape dimensions equals numel.
pub fn require_shape_product(shape: &[usize], numel: usize, context: &str) -> Result<(), String> {
    let product: usize = shape.iter().product();
    if product != numel {
        Err(format!(
            "Shape product mismatch in {}: expected numel={}, product={}",
            context, numel, product
        ))
    }
    Ok(())
}

/// Validates that a dimension is within bounds.
pub fn require_dim_bound(dim: usize, max: usize, axis: usize, context: &str) -> Result<(), String> {
    if dim >= max {
        Err(format!("Dimension {} out of bounds in {}: {} >= {}", axis, context, dim, max))
    } else {
        Ok(())
    }
}

/// Validates that a value is finite.
pub fn require_finite(value: f64, context: &str) -> Result<(), String> {
    if value.is_nan() {
        return Err(format!("NaN detected in {}", context));
    }
    if value.is_infinite() {
        return Err(format!("Inf detected in {}", context));
    }
    Ok(())
}

/// Validates that a value is non-negative.
pub fn require_nonneg(value: f64, context: &str) -> Result<(), String> {
    if value < 0.0 {
        Err(format!("Negative value {} in {}", value, context))
    } else {
        Ok(())
    }
}

/// Validates that a value is positive (strictly greater than zero).
pub fn require_positive(value: f64, context: &str) -> Result<(), String> {
    if value <= 0.0 {
        Err(format!("Non-positive value {} in {}", value, context))
    } else {
        Ok(())
    }
}

/// Validates that a tensor is 2D and square.
pub fn require_square_matrix(tensor: &Tensor, context: &str) -> Result<(), String> {
    if !tensor.is_matrix() {
        return Err(format!("Expected matrix in {}", context));
    }
    let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
    if rows != cols {
        return Err(format!("Expected square matrix in {}, got {}x{}", context, rows, cols));
    }
    Ok(())
}

/// Validates that a tensor has the expected rank.
pub fn require_ndim(tensor: &Tensor, expected: usize, context: &str) -> Result<(), String> {
    let actual = tensor.ndim();
    if actual != expected {
        Err(format!(
            "Expected {}-dimensional tensor in {}, got {}-dimensional",
            expected, context, actual
        ))
    } else {
        Ok(())
    }
}

/// Validates that a tensor has at least the minimum required rank.
pub fn require_min_ndim(tensor: &Tensor, min: usize, context: &str) -> Result<(), String> {
    if tensor.ndim() < min {
        Err(format!(
            "Expected at least {}-dimensional tensor in {}, got {}-dimensional",
            min, context, tensor.ndim()
        ))
    } else {
        Ok(())
    }
}

// =============================================================================
// String Formatting Utilities
// =============================================================================

/// Formats a byte count as a human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 { return format!("{} B", bytes); }
    if bytes < 1024 * 1024 { return format!("{:.1} KB", bytes as f64 / 1024.0); }
    if bytes < 1024 * 1024 * 1024 { return format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0)); }
    if bytes < 1024usize.pow(4) { return format!("{:.1} GB", bytes as f64 / (1024.0_f64.pow(3))); }
    format!("{:.1} TB", bytes as f64 / (1024.0_f64.pow(4)))
}

/// Formats a shape as a human-readable string.
pub fn format_shape(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({})", shape[0]),
        2 => format!("({}, {})", shape[0], shape[1]),
        _ => format!("({})", shape.iter().join(", "x")),
    }
}

/// Formats a number of elements with appropriate SI prefixes.
pub fn format_numel(n: usize) -> String {
    if n < 1000 { return format!("{}", n); }
    if n < 1_000_000 { return format!("{:.1}K", n as f64 / 1000.0); }
    if n < 1_000_000_000 { return format!("{:.1}M", n as f64 / 1_000_000.0); }
    if n < 1_000_000_000_000 { return format!("{:.1}B", n as f64 / 1_000_000_000.0); }
    format!("{:.1}T", n as f64 / 1_000_000_000_000.0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), "0.1.0");
    }

    #[test]
    fn test_git_hash() {
        assert!(!git_hash().is_empty());
    }

    #[test]
    fn test_version_string() {
        let s = version_string();
        assert!(s.contains("0.1.0"));
        assert!(s.contains("brain-core"));
    }

    #[test]
    fn test_version_tuple() {
        assert_eq!(version_tuple(), (0, 1, 0));
    }

    #[test]
    fn test_rustc_version() {
        assert!(!rustc_version().is_empty());
    }

    #[test]
    fn test_target_arch() {
        assert!(!target_arch().is_empty());
    }

    #[test]
    fn test_target_os() {
        assert!(!target_os().is_empty());
    }

    #[test]
    fn test_module_count() {
        assert_eq!(module_count(), 9);
    }

    #[test]
    fn test_source_lines() {
        assert!(source_lines() > 0);
    }

    #[test]
    fn test_framework_size_bytes() {
        assert!(framework_size_bytes() > 0);
    }

    #[test]
    fn test_module_info() {
        let info = module_info();
        assert!(info.contains("error.rs"));
        assert!(info.contains("tensor"));
        assert!(info.contains("lib.rs"));
    }

    #[test]
    fn test_build_info() {
        let info = build_info();
        assert!(info.contains("brain-core"));
        assert!(info.contains("Rust"));
    }

    #[test]
    fn test_initialize() {
        assert!(!is_initialized());
        initialize();
        assert!(is_initialized());
    }

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert!(!config.debug);
        assert_eq!(config.seed, 42);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = Config::new()
            .with_debug(true)
            .with_seed(123)
            .with_max_threads(4);
        assert!(config.debug);
        assert_eq!(config.seed, 123);
        assert_eq!(config.max_threads, 4);
    }

    #[test]
    fn test_config_summary() {
        let config = Config::default();
        let summary = config.summary();
        assert!(summary.contains("debug: false"));
        assert!(summary.contains("cpu"));
    }

    #[test]
    fn test_config_set_get() {
        let config = Config::new().with_seed(999);
        set_config(config).unwrap();
        let loaded = config();
        assert_eq!(loaded.seed, 999);
    }

    #[test]
    fn test_config_with_config() {
        let result = with_config(|c| {
            c.seed = 777;
            c.debug = true;
            777
        });
        assert_eq!(result, 777);
        let config = config();
        assert_eq!(config.seed, 777);
        assert!(config.debug);
    }

    #[test]
    fn test_check_version_compat() {
        assert!(check_version_compat("0.1.0").is_ok());
        assert!(check_version_compat("0.0.0").is_ok());
        assert!(check_version_compat("0.2.0").is_err());
    }

    #[test]
    fn test_prelude() {
        use prelude::*;
        let _err: BrainError = BrainError::invalid_value("test");
        let _dtype: DType = DType::F32;
        let _device: Device = Device::Cpu;
        let _shape: Shape = Shape::from_dims(&[2, 3]);
        let _version: &str = VERSION;
    }

    #[test]
    fn test_type_aliases() {
        let _: OwnedTensor = Tensor::ones(vec![2, 3]);
        let _: Vector = Tensor::zeros(vec![5]);
        let _: Matrix = Tensor::identity(3);
        let _: Tensor3D = Tensor::zeros(vec![2, 3, 4]);
        let _: BatchTensor = Tensor::zeros(vec![1, 3, 4]);
        let _: Scalar = Tensor::scalar(1.0);
    }

    #[test]
    fn test_tensors_close() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[1.0001, 2.0002, 3.0003], vec![3]);
        assert!(tensors_close(&a, &b, 0.01, 0.0));
    }

    #[test]
    fn test_tensors_close_different() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3]);
        assert!(!tensors_close(&a, &b, 0.01, 0.0));
    }

    #[test]
    fn test_tensors_close_different_shapes() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        assert!(!tensors_close(&a, &b, 0.01, 0.0));
    }

    #[test]
    fn test_count_differences() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        assert_eq!(count_differences(&a, &b), 0);

        let c = Tensor::from_slice(&[1.0, 2.0, 3.5], vec![3]);
        assert_eq!(count_differences(&a, &c), 1);
    }

    #[test]
    fn test_max_difference() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 5.0], vec![3]);
        assert!((max_difference(&a, &b) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_require_same_shape() {
        assert!(require_same_shape(&[2, 3], &[2, 3], "test").is_ok());
        assert!(require_same_shape(&[2, 3], &[3, 2], "test").is_err());
    }

    #[test]
    fn test_require_positive_dims() {
        assert!(require_positive_dims(&[2, 3], "test").is_ok());
        assert!(require_positive_dims(&[0, 3], "test").is_err());
    }

    #[test]
    fn test_require_shape_product() {
        assert!(require_shape_product(&[2, 3], 6, "test").is_ok());
        assert!(require_shape_product(&[2, 3], 5, "test").is_err());
    }

    #[test]
    fn test_require_dim_bound() {
        assert!(require_dim_bound(0, 5, 0, "test").is_ok());
        assert!(require_dim_bound(5, 5, 0, "test").is_err());
    }

    #[test]
    fn test_require_finite() {
        assert!(require_finite(1.0, "test").is_ok());
        assert!(require_finite(f64::NAN, "test").is_err());
        assert!(require_finite(f64::INFINITY, "test").is_err());
    }

    #[test]
    fn test_require_nonneg() {
        assert!(require_nonneg(0.0, "test").is_ok());
        assert!(require_nonneg(-1.0, "test").is_err());
    }

    #[test]
    fn test_require_positive() {
        assert!(require_positive(1.0, "test").is_ok());
        assert!(require_positive(0.0, "test").is_err());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }

    #[test]
    fn test_format_shape() {
        assert_eq!(format_shape(&[]), "()");
        assert_eq!(format_shape(&[5]), "(5)");
        assert_eq!(format_shape(&[2, 3]), "(2, 3)");
        assert_eq!(format_shape(&[1, 2, 3, 4]), "(1, 2, 3, 4)");
    }

    #[test]
    fn test_format_numel() {
        assert_eq!(format_numel(0), "0");
        assert_eq!(format_numel(500), "500");
        assert_eq!(format_numel(1500), "1.5K");
        assert_eq!(format_numel(2000000), "2.0M");
    }

    #[test]
    fn test_vec_from_values() {
        let v = vec_from_values(&[1.0, 2.0, 3.0]);
        assert_eq!(v.shape(), &[3]);
        assert_eq!(v.get(0), 1.0);
        assert_eq!(v.get(2), 3.0);
    }

    #[test]
    fn test_matrix_from_values() {
        let m = matrix_from_values(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(m.shape(), &[2, 2]);
        assert_eq!(m.get_index(&[0, 0]), 1.0);
        assert_eq!(m.get_index(&[1, 1]), 4.0);
    }

    #[test]
    fn test_diag_from_values() {
        let d = diag_from_values(&[1.0, 2.0, 3.0]);
        assert_eq!(d.shape(), &[3, 3]);
        assert_eq!(d.get_index(&[0, 0]), 1.0);
        assert_eq!(d.get_index(&[1, 1]), 2.0);
    }

    #[test]
    fn test_triu_from_values() {
        let m = triu_from_values(&[1.0, 2.0, 3.0], 3);
        assert_eq!(m.get_index(&[0, 0]), 1.0);
        assert_eq!(m.get_index(&[1, 1]), 2.0);
        assert_eq!(m.get_index(&[2, 2]), 3.0);
    }

    #[test]
    fn test_tril_from_values() {
        let m = tril_from_values(&[1.0, 2.0, 3.0], 3);
        assert_eq!(m.get_index(&[1, 0]), 2.0);
        assert_eq!(m.get_index(&[2, 1]), 3.0);
    }

    #[test]
    fn test_symmetric_from_values() {
        let m = symmetric_from_values(&[1.0, 2.0, 3.0], 3);
        assert_eq!(m.get_index(&[0, 1]), 2.0);
        assert_eq!(m.get_index(&[1, 0]), 2.0);
    }

    #[test]
    fn test_randn() {
        let t = randn(vec![100, 100], 0.0, 1.0);
        assert_eq!(t.shape(), &[100, 100]);
        let stats = t.statistics();
        assert!(stats.mean.abs() < 0.5);
        assert!(stats.std.abs() < 0.5);
    }

    #[test]
    fn test_randu() {
        let t = randu(vec![100, 100]);
        assert_eq!(t.shape(), &[100, 100]);
        let stats = t.statistics();
        assert!(stats.min >= 0.0);
        assert!(stats.max <= 1.0);
    }

    #[test]
    fn test_zeros_like() { assert!(zeros_like(&[2, 3]).is_empty()); }
    #[test]
    fn test_ones_like() { assert_eq!(ones_like(&[2, 3]).get(5), 1.0); }
    #[test]
    fn test_full_like() { assert_eq!(full_like(&[2, 3], 5.0).get(0), 5.0); }
    #[test]
    fn test_eye_like() { assert_eq!(eye_like(3).get_index(&[0, 0]), 1.0); }

    #[test]
    fn test_mlp_shape() {
        let shapes = mlp_shape(&[10, 20, 5]);
        assert_eq!(shapes.len(), 2);
        assert_eq!(shapes[0], vec![10, 20]);
        assert_eq!(shapes[1], vec![20, 5]);
    }

    #[test]
    fn test_count_parameters() {
        let shapes = mlp_shape(&[784, 256, 10]);
        assert_eq!(count_parameters(&shapes), 784 * 256 + 256 * 10);
    }

    #[test]
    fn test_compute_macs() {
        let shapes = mlp_shape(&[100, 200, 50]);
        assert_eq!(compute_macs(&shapes), 100 * 200 + 200 * 50);
    }

    #[test]
    fn test_compute_flops() {
        let shapes = mlp_shape(&[100, 200, 50]);
        let macs = compute_macs(&shapes);
        let flops = compute_flops(&shapes, 2);
        assert_eq!(flops, macs * 2);
    }

    #[test]
    fn test_conv_output() {
        let output = conv_output(&[1, 3, 32, 32], &[3, 3], &[1, 1], &[0, 0]);
        assert_eq!(output, vec![1, 3, 32, 32]);
    }

    #[test]
    fn test_linear_output() {
        let output = linear_output(&[8, 32], 10);
        assert_eq!(output, vec![8, 32, 10]);
    }

    #[test]
    fn test_require_ndim() {
        let t = Tensor::zeros(vec![2, 3]);
        assert!(require_ndim(&t, 2, "test").is_ok());
        assert!(require_ndim(&t, 3, "test").is_err());
    }

    #[test]
    fn test_require_min_ndim() {
        let t = Tensor::zeros(vec![2, 3]);
        assert!(require_min_ndim(&t, 2, "test").is_ok());
        assert!(require_min_ndim(&t, 3, "test").is_ok());
        assert!(require_min_ndim(&t, 4, "test").is_err());
    }

    #[test]
    fn test_require_square_matrix() {
        let a = Tensor::identity(3);
        assert!(require_square_matrix(&a, "test").is_ok());
        let b = Tensor::zeros(vec![2, 3]);
        assert!(require_square_matrix(&b, "test").is_err());
    }

    #[test]
    fn test_deprecation_warn() {
        deprecation_warn("old_api");
    }

    #[test]
    fn test_debug_log() {
        let config = Config::new().with_debug(true);
        set_config(config).ok();
        debug_log("test message");
    }

    #[test]
    fn test_perf_warn() {
        perf_warn("using slow fallback");
    }
}
