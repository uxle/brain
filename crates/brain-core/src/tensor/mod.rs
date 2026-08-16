//! Tensor module - core data structures and operations for the Brain deep learning framework.
//!
//! This module provides the [`Tensor`] struct, which is the central data structure
//! for all numerical computations in the framework. Tensors are multi-dimensional
//! arrays with support for automatic differentiation, device placement, and
//! various memory layouts.
//!
//! # Module Organization
//!
//! * `impl` - Core `Tensor` struct implementation
//! * `arithmetic` - Arithmetic operations (add, sub, mul, div, matmul, etc.)
//! * `math` - Mathematical functions (sin, cos, exp, log, activation functions)
//! * `linalg` - Linear algebra operations (norm, det, inv, svd, etc.)
//! * `reduction` - Reduction operations (sum, mean, min, max, etc.)
//! * `indexing` - Advanced indexing operations

pub mod arithmetic;
pub mod math;
pub mod linalg;
pub mod reduction;
pub mod indexing;

#[path = "impl.rs"]
mod impl_tensor;

pub use impl_tensor::*;

// =============================================================================
// TensorStats
// =============================================================================

/// Statistics computed over a tensor's elements.
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Minimum element value.
    pub min: f64,
    /// Maximum element value.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Number of zero elements.
    pub num_zeros: usize,
    /// Number of NaN elements.
    pub num_nans: usize,
    /// Sparsity ratio (num_zeros / total_elements).
    pub sparsity: f64,
    /// L1 norm (sum of absolute values).
    pub l1_norm: f64,
    /// L2 norm (Euclidean norm).
    pub l2_norm: f64,
}

impl Default for TensorStats {
    fn default() -> Self {
        TensorStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            num_zeros: 0,
            num_nans: 0,
            sparsity: 0.0,
            l1_norm: 0.0,
            l2_norm: 0.0,
        }
    }
}

impl std::fmt::Display for TensorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TensorStats:")?;
        writeln!(f, "  min:      {:.6}", self.min)?;
        writeln!(f, "  max:      {:.6}", self.max)?;
        writeln!(f, "  mean:     {:.6}", self.mean)?;
        writeln!(f, "  std:      {:.6}", self.std)?;
        writeln!(f, "  zeros:    {}", self.num_zeros)?;
        writeln!(f, "  nans:     {}", self.num_nans)?;
        writeln!(f, "  sparsity: {:.4}%", self.sparsity * 100.0)?;
        writeln!(f, "  l1_norm:  {:.6}", self.l1_norm)?;
        write!(f, "  l2_norm:  {:.6}", self.l2_norm)
    }
}

// =============================================================================
// TensorIter
// =============================================================================

/// An iterator over the elements of a tensor in row-major order.
pub struct TensorIter<'a> {
    data: &'a [f64],
    pos: usize,
}

impl<'a> TensorIter<'a> {
    pub fn new(data: &'a [f64]) -> Self {
        TensorIter { data, pos: 0 }
    }
}

impl<'a> Iterator for TensorIter<'a> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.data.len() {
            let val = self.data[self.pos];
            self.pos += 1;
            Some(val)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for TensorIter<'a> {}

// =============================================================================
// TensorIterMut
// =============================================================================

/// A mutable iterator over the elements of a tensor in row-major order.
pub struct TensorIterMut<'a> {
    data: &'a mut [f64],
    pos: usize,
}

impl<'a> TensorIterMut<'a> {
    pub fn new(data: &'a mut [f64]) -> Self {
        TensorIterMut { data, pos: 0 }
    }
}

impl<'a> Iterator for TensorIterMut<'a> {
    type Item = &'a mut f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.data.len() {
            let val = &mut self.data[self.pos];
            self.pos += 1;
            Some(val)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for TensorIterMut<'a> {}

// =============================================================================
// TensorIndex
// =============================================================================

/// Represents an index specification for tensor indexing operations.
#[derive(Debug, Clone)]
pub enum TensorIndex {
    /// A single integer index.
    Single(isize),
    /// A slice with optional start, end, and step.
    Slice {
        start: Option<isize>,
        end: Option<isize>,
        step: isize,
    },
    /// Ellipsis to fill remaining dimensions.
    Ellipsis,
    /// New axis insertion.
    NewAxis,
    /// Boolean mask indexing.
    Boolean(Vec<bool>),
    /// Advanced (fancy) indexing with a list of indices.
    Advanced(Vec<isize>),
}

impl TensorIndex {
    /// Creates an index from a single integer.
    pub fn single(i: isize) -> Self {
        TensorIndex::Single(i)
    }

    /// Creates a slice from start to end with step 1.
    pub fn slice(start: isize, end: isize) -> Self {
        TensorIndex::Slice { start: Some(start), end: Some(end), step: 1 }
    }

    /// Creates a full-range slice.
    pub fn all() -> Self {
        TensorIndex::Slice { start: None, end: None, step: 1 }
    }

    /// Creates an ellipsis index.
    pub fn ellipsis() -> Self {
        TensorIndex::Ellipsis
    }

    /// Creates a new axis index.
    pub fn new_axis() -> Self {
        TensorIndex::NewAxis
    }

    /// Creates a boolean mask index.
    pub fn boolean(mask: Vec<bool>) -> Self {
        TensorIndex::Boolean(mask)
    }

    /// Creates an advanced (fancy) index.
    pub fn advanced(indices: Vec<isize>) -> Self {
        TensorIndex::Advanced(indices)
    }

    /// Resolves the index given a dimension size.
    pub fn resolve(&self, dim_size: usize) -> (usize, usize, usize, usize) {
        match self {
            TensorIndex::Single(i) => {
                let idx = if *i < 0 { dim_size as isize + i } else { *i };
                let idx = idx.max(0).min(dim_size as isize - 1) as usize;
                (idx, idx + 1, 1, 0)
            }
            TensorIndex::Slice { start, end, step } => {
                let abs_step = if *step >= 0 { *step as usize } else { (-(*step)) as usize };
                let s = match start {
                    Some(v) => {
                        if *v < 0 { (dim_size as isize + v).max(0) as usize }
                        else { (*v as usize).min(dim_size) }
                    }
                    None => if *step < 0 { dim_size } else { 0 },
                };
                let e = match end {
                    Some(v) => {
                        if *v < 0 { (dim_size as isize + v).max(0) as usize }
                        else { (*v as usize).min(dim_size) }
                    }
                    None => if *step < 0 { 0 } else { dim_size },
                };
                let len = if s > e { 0 } else { (e - s + abs_step - 1) / abs_step };
                (s, e, abs_step, len)
            }
            _ => (0, dim_size, 1, dim_size),
        }
    }
}

// =============================================================================
// Layout Enum
// =============================================================================

/// The memory layout of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major (C-style) layout.
    RowMajor,
    /// Column-major (Fortran-style) layout.
    ColMajor,
    /// Sparse layout.
    Sparse,
    /// Blocked layout.
    Blocked { block_size: usize },
}

impl Default for Layout {
    fn default() -> Self {
        Layout::RowMajor
    }
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Layout::RowMajor => write!(f, "row_major"),
            Layout::ColMajor => write!(f, "col_major"),
            Layout::Sparse => write!(f, "sparse"),
            Layout::Blocked { block_size } => write!(f, "blocked({})", block_size),
        }
    }
}

// =============================================================================
// TensorFlags
// =============================================================================

/// Bitflags for tensor properties and optimization hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorFlags(u32);

impl TensorFlags {
    pub const NONE: TensorFlags = TensorFlags(0);
    pub const REQUIRES_GRAD: TensorFlags = TensorFlags(1 << 0);
    pub const CONTIGUOUS: TensorFlags = TensorFlags(1 << 1);
    pub const IS_VIEW: TensorFlags = TensorFlags(1 << 2);
    pub const READ_ONLY: TensorFlags = TensorFlags(1 << 3);
    pub const PINNED: TensorFlags = TensorFlags(1 << 4);
    pub const GRAD_DIRTY: TensorFlags = TensorFlags(1 << 5);

    pub fn contains(&self, flag: TensorFlags) -> bool {
        (self.0 & flag.0) != 0
    }

    pub fn set(&mut self, flag: TensorFlags) {
        self.0 |= flag.0;
    }

    pub fn unset(&mut self, flag: TensorFlags) {
        self.0 &= !flag.0;
    }

    pub fn toggle(&mut self, flag: TensorFlags) {
        self.0 ^= flag.0;
    }

    pub fn bits(&self) -> u32 { self.0 }

    pub fn is_empty(&self) -> bool { self.0 == 0 }
}

impl Default for TensorFlags {
    fn default() -> Self { TensorFlags::CONTIGUOUS }
}

impl std::ops::BitOr for TensorFlags {
    type Output = TensorFlags;
    fn bitor(self, rhs: TensorFlags) -> TensorFlags { TensorFlags(self.0 | rhs.0) }
}

impl std::ops::BitOrAssign for TensorFlags {
    fn bitor_assign(&mut self, rhs: TensorFlags) { self.0 |= rhs.0; }
}

// =============================================================================
// Pretty Printing
// =============================================================================

/// Formats a flat slice of f64 values as a pretty-printed string with the given shape.
pub fn pretty_print(data: &[f64], shape: &[usize], indent: usize) -> String {
    if shape.is_empty() {
        return format!("{:.6}", data.first().copied().unwrap_or(0.0));
    }
    if shape.len() == 1 {
        let vals: Vec<String> = data.iter().take(shape[0]).map(|v| format!("{:8.4}", v)).collect();
        let padding = " ".repeat(indent);
        return format!("{}[{}]", padding, vals.join(", "));
    }
    let padding = " ".repeat(indent);
    let mut lines = Vec::new();
    lines.push(format!("{}[", padding));
    let sub_dim = &shape[1..];
    let sub_size: usize = sub_dim.iter().product();
    let outer_dim = shape[0].min(10);
    for i in 0..outer_dim {
        let start = i * sub_size;
        let end = start + sub_size;
        let sub_data = if start < data.len() { &data[start..end.min(data.len())] } else { &[] };
        lines.push(pretty_print(sub_data, sub_dim, indent + 2));
    }
    if shape[0] > 10 { lines.push(format!("{},", " ".repeat(indent + 2))); }
    let trailing_comma = if shape[0] > 1 { "," } else { "" };
    lines.push(format!("{}{}{}", padding, trailing_comma, "]"));
    lines.join("\n")
}

// =============================================================================
// Shape validation helpers
// =============================================================================

/// Validates shapes are compatible for element-wise binary operations.
pub fn validate_binary_shapes(a_shape: &[usize], b_shape: &[usize]) -> bool {
    let max_ndim = a_shape.len().max(b_shape.len());
    for i in 0..max_ndim {
        let da = if i < a_shape.len() { a_shape[a_shape.len() - 1 - i] } else { 1 };
        let db = if i < b_shape.len() { b_shape[b_shape.len() - 1 - i] } else { 1 };
        if da != db && da != 1 && db != 1 { return false; }
    }
    true
}

/// Computes broadcast output shape for two tensors.
pub fn binary_broadcast_shape(a_shape: &[usize], b_shape: &[usize]) -> Vec<usize> {
    let max_ndim = a_shape.len().max(b_shape.len());
    let mut result = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        let da = if i < a_shape.len() { a_shape[a_shape.len() - 1 - i] } else { 1 };
        let db = if i < b_shape.len() { b_shape[b_shape.len() - 1 - i] } else { 1 };
        result.push(if da == 1 { db } else { da });
    }
    result.reverse();
    result
}

/// Computes the flat index for a broadcast-access pattern.
pub fn broadcast_flat_index(output_idx: usize, output_shape: &[usize], source_shape: &[usize]) -> usize {
    let output_ndim = output_shape.len();
    let source_ndim = source_shape.len();
    let rank_diff = output_ndim - source_ndim;
    let mut multi = vec![0usize; output_ndim];
    let mut idx = output_idx;
    for i in (0..output_ndim).rev() {
        if output_shape[i] > 0 {
            multi[i] = idx % output_shape[i];
            idx /= output_shape[i];
        }
    }
    let mut source_multi = Vec::with_capacity(source_ndim);
    for i in 0..source_ndim {
        let src_dim = source_shape[i];
        if src_dim == output_shape[rank_diff + i] {
            source_multi.push(multi[rank_diff + i]);
        } else {
            source_multi.push(0);
        }
    }
    let mut flat = 0;
    let mut stride = 1;
    for i in (0..source_ndim).rev() {
        flat += source_multi[i] * stride;
        stride *= source_shape[i];
    }
    flat
}

/// Computes strides for row-major layout given a shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 { return vec![]; }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Converts a flat index to multi-dimensional indices given shape and strides.
pub fn flat_index_to_multi(flat: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut idx = flat;
    let ndim = shape.len();
    let mut multi = vec![0usize; ndim];
    for i in 0..ndim {
        if strides[i] > 0 {
            multi[i] = idx / strides[i];
            idx %= strides[i];
        }
    }
    multi
}

/// Converts multi-dimensional indices to a flat linear index.
pub fn multi_index_to_flat(multi: &[usize], strides: &[usize]) -> usize {
    let mut flat = 0;
    for (i, &idx) in multi.iter().enumerate() {
        if i < strides.len() { flat += idx * strides[i]; }
    }
    flat
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_stats_default() {
        let stats = TensorStats::default();
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.num_zeros, 0);
    }

    #[test]
    fn test_tensor_stats_display() {
        let stats = TensorStats {
            min: -1.0, max: 2.0, mean: 0.5, std: 0.7,
            num_zeros: 10, num_nans: 0, sparsity: 0.1,
            l1_norm: 5.0, l2_norm: 3.0,
        };
        let display = format!("{}", stats);
        assert!(display.contains("min:"));
    }

    #[test]
    fn test_tensor_iter() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let iter = TensorIter::new(&data);
        assert_eq!(iter.sum::<f64>(), 10.0);
    }

    #[test]
    fn test_tensor_iter_empty() {
        let data: Vec<f64> = vec![];
        assert_eq!(TensorIter::new(&data).count(), 0);
    }

    #[test]
    fn test_tensor_iter_mut() {
        let mut data = vec![1.0, 2.0, 3.0];
        for v in TensorIterMut::new(&mut data) { *v *= 2.0; }
        assert_eq!(data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_tensor_index_single() {
        let idx = TensorIndex::single(5);
        let (start, end, step, len) = idx.resolve(10);
        assert_eq!(start, 5);
        assert_eq!(end, 6);
        assert_eq!(step, 1);
    }

    #[test]
    fn test_tensor_index_negative() {
        let (start, _, _, _) = TensorIndex::single(-1).resolve(10);
        assert_eq!(start, 9);
    }

    #[test]
    fn test_tensor_index_slice() {
        let (start, end, step, len) = TensorIndex::slice(2, 7).resolve(10);
        assert_eq!(start, 2);
        assert_eq!(end, 7);
        assert_eq!(step, 1);
        assert_eq!(len, 5);
    }

    #[test]
    fn test_layout_default() {
        assert_eq!(Layout::default(), Layout::RowMajor);
    }

    #[test]
    fn test_tensor_flags() {
        let mut flags = TensorFlags::NONE;
        assert!(flags.is_empty());
        flags.set(TensorFlags::REQUIRES_GRAD);
        assert!(flags.contains(TensorFlags::REQUIRES_GRAD));
        flags.toggle(TensorFlags::CONTIGUOUS);
        assert!(flags.contains(TensorFlags::CONTIGUOUS));
    }

    #[test]
    fn test_tensor_flags_bitor() {
        let c = TensorFlags::REQUIRES_GRAD | TensorFlags::CONTIGUOUS;
        assert!(c.contains(TensorFlags::REQUIRES_GRAD));
        assert!(c.contains(TensorFlags::CONTIGUOUS));
    }

    #[test]
    fn test_validate_binary_shapes() {
        assert!(validate_binary_shapes(&[2, 3], &[2, 3]));
        assert!(validate_binary_shapes(&[2, 1], &[1, 3]));
        assert!(!validate_binary_shapes(&[2, 3], &[4, 5]));
    }

    #[test]
    fn test_binary_broadcast_shape() {
        assert_eq!(binary_broadcast_shape(&[2, 1], &[1, 3]), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_flat_index() {
        assert_eq!(broadcast_flat_index(0, &[2, 3], &[2, 1]), 0);
        assert_eq!(broadcast_flat_index(3, &[2, 3], &[2, 1]), 1);
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_flat_multi_roundtrip() {
        let shape = vec![2, 3, 4];
        let strides = compute_strides(&shape);
        for i in 0..24 {
            let multi = flat_index_to_multi(i, &shape, &strides);
            let back = multi_index_to_flat(&multi, &strides);
            assert_eq!(back, i);
        }
    }

    #[test]
    fn test_pretty_print_1d() {
        let data = vec![1.0, 2.0, 3.0];
        let s = pretty_print(&data, &[3], 0);
        assert!(s.contains("1.0"));
    }

    #[test]
    fn test_pretty_print_scalar() {
        let data = vec![42.0];
        let s = pretty_print(&data, &[], 0);
        assert!(s.contains("42"));
    }
}
